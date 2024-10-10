#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

# ML functions
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree

# Custom libraries for working with climate
import climperiods
from ersst5tools import ERSST5
from era5tools import ERA5
from seas5tools import SEAS5


class TeleSST():
    def __init__(self, lat_range=(-90,90), weighting='rootcoslat'):
        """Class constructor to generate teleconnection features from SSTs.

        Parameters
        ----------
            lat_range : (float, float), optional
                Latitude range subset to use.
            weighting : str, optional
                Weighting to apply to cells before processing. Either
                  'rootcoslat' - sqrt(cosine(latitude)) [default] or
                  'coslat' - cosine(latitude)
        """

        self.lat_range = lat_range
        self.weighting = weighting

        self.now = datetime.datetime.now()
        self.EOFs = None
        self.PCs = None
        self.varexp = None

    def calc_anoms(self, inpath, dataset, year_range):
        """Load SST data and calculate anomalies.

        Parameters
        ----------
            inpath : str
                Path to raw SST data.
            dataset : str
                Which dataset to use - must be either ERA5 or ERSST5.
            year_range : (int, int)
                Year range to process.
        """

        if year_range[1] > self.now.year:
            print(f'year range {year_range} cannot include the future')
            return None

        # Define the climate periods used to calculate the anomalies
        self.clims = climperiods.clims(*year_range)

        # Load the anomalies
        if dataset.lower() == 'era5':
            era5 = ERA5(cdsapi_key='dummy')
            self.anoms = era5.calc_anoms(inpath, 'sst', year_range)
        elif dataset.lower() == 'ersst5':
            ersst5 = ERSST5()
            self.anoms = ersst5.calc_anoms(inpath, year_range
                                           ).rename({'lat': 'latitude',
                                                     'lon': 'longitude'})
        else:
            print('dataset must be one of ERA5 or ERSST5')
            return None

    def calc_anoms_forecast(self, inpath, fyear, fmonth, clim_year_range=(1993, 2016)):
        """Load forecast SEAS5 SST data and calculate anomalies.

        Parameters
        ----------
            inpath : str
                Path to raw forecast SST data.
            fyear : int
                Year of forecast.
            fmonth : int
                Month of forecast.
            clim_year_range : (int, int), optional
                Year range of climatology used to calculate anomalies.
                Defaults to (1993, 2016) per SEAS5 documentation. If a
                different range is specified, it must first be calculated
                in the seas5tools package.
        """

        if (fyear > self.now.year or
            (fyear == self.now.year and fmonth > self.now.month)):
            print(f'Forecast year/month cannot be in the future')
            return None
        seas5 = SEAS5(cdsapi_key='dummy')
        self.anoms_fore = seas5.calc_anoms(inpath, 'sst', fyear, fmonth,
                                           clim_year_range)

    def fit(self, da):
        """Calculate EOFs and PCs of reference data at monthly resolution.

        Parameters
        ----------
            da : DataArray
                DataArray with dims including
                ['latitude','longitude','year','month'].
        """

        # Subset latitudes
        da = da.sel(latitude=slice(*self.lat_range)) * 1

        # Calculate weights
        if self.weighting == 'coslat':
            self.wts_da = np.cos(np.deg2rad(da['latitude']))
        elif self.weighting == 'rootcoslat':
            self.wts_da = np.sqrt(np.cos(np.deg2rad(da['latitude'])))
        else:
            self.wts_da = xr.DataArray(1)
        self.wts_ss = self.wts_da.to_series()

        # Calculate EOFs and PCs for each month
        EOFs, PCs, varexp = {}, {}, {}
        pca = PCA()
        for m in range(1, 13):
            da_month = da.sel(month=m) * self.wts_da
            X = da_month.to_series().dropna().unstack(['latitude','longitude'])
            pca.fit(X)
            EOFs[m] = pd.DataFrame(pca.components_, columns=X.columns)
            PCs[m] = X @ EOFs[m].T

            # Orient EOFs of successive months consistently for ease of interpretation
            if m > 1:
                ix = EOFs[m].columns.intersection(EOFs[m-1].columns)
                sgn = np.sign(np.diag(EOFs[m][ix] @ EOFs[m-1][ix].T))
                EOFs[m] = EOFs[m] * sgn[:,None]
                PCs[m] = PCs[m] * sgn[None,:]

            varexp[m] = pca.explained_variance_ratio_

        # Convert to DataFrames
        self.EOFs = pd.concat(EOFs, names=['month','pc'])
        self.PCs = pd.concat(PCs, names=['month']
                             ).reorder_levels(['year','month']
                                              ).sort_index().rename_axis('pc', axis=1)
        self.varexp = pd.DataFrame(varexp).rename_axis('month', axis=1
                                                       ).rename_axis('pc', axis=0)

    def to_file(self, outpath, desc):
        """Save EOFs and PCs to disk.

        Parameters
        ----------
            outpath : str
                Output path.
            desc : str
                Description of model.
        """
        self.EOFs.to_parquet(os.path.join(outpath, f'EOFs_{desc}.parquet'))
        self.PCs.to_parquet(os.path.join(outpath, f'PCs_{desc}.parquet'))
        self.varexp.to_parquet(os.path.join(outpath, f'varexp_{desc}.parquet'))

    def from_file(self, inpath, desc):
        """Load model from disk.

        Parameters
        ----------
            inpath : str
                Input path.
            desc : str
                Description of model.
        """
        self.EOFs = pd.read_parquet(os.path.join(inpath, f'EOFs_{desc}.parquet'))
        self.PCs = pd.read_parquet(os.path.join(inpath, f'PCs_{desc}.parquet'))
        self.varexp = pd.read_parquet(os.path.join(inpath, f'varexp_{desc}.parquet'))

        # Calculate weights
        lats = self.EOFs.columns.unique(level='latitude').to_series()
        if self.weighting == 'coslat':
            self.wts_ss = np.cos(np.deg2rad(lats))
        elif self.weighting == 'rootcoslat':
            self.wts_ss = np.sqrt(np.cos(np.deg2rad(lats)))
        else:
            self.wts_ss = (lats/lats).fillna(1)
        self.wts_da = self.wts_ss.to_xarray()

    def project(self, da, forecast=True):
        """Project new data onto EOFs previously fitted to get new PCs.

        Parameters
        ----------
            da : DataArray
                DataArray with dims including
                ['latitude','longitude','year','month'].
            forecast : bool
                Flag if da is a forecast or not. Makes standard assumptions
                about the structure of a processed SEAS5 forecast.

        Returns
        -------
            PCs_proj : DataFrame
                Projected PCs.
        """

        # Subset latitudes
        da = da.sel(latitude=slice(*self.lat_range)) * 1

        # Extract all non-null lat/lons and convert to radians
        das = da.stack(latlon=['latitude','longitude'])
        ll = ~das.isnull().all(tuple(set(das.dims) - {'latlon'}))
        X = np.deg2rad(ll[ll].get_index('latlon').to_frame())

        # Look up the indices of non-null cells nearest to EOF cells
        balltree = BallTree(X, metric='haversine')
        ix = balltree.query(np.deg2rad(self.EOFs.columns.to_frame()),
                            return_distance=False).ravel()

        if forecast:
            fmonth = int(da['fmonth'])
            fyear = int(da['year'].min()) if fmonth < 12 else int(da['year'])-1
            lead_months = np.arange(1, 7)
            months = (fmonth + lead_months - 1) % 12 + 1
            years = fyear + (fmonth + lead_months - 1) // 12
            ym = zip(years, months)
        else:
            ym = da.stack(ym=['year','month']).ym.values

        PCs_proj = {}
        for year, month in ym:
            if forecast:
                to_proj = da.sel(year=year, month=month
                                 ).to_series().dropna().unstack('number')
            else:
                to_proj = da.sel(month=month).to_series().dropna().unstack('year')
            to_proj_nearest = pd.DataFrame(to_proj.to_numpy()[ix],
                                           index=self.EOFs.columns,
                                           columns=to_proj.columns).sort_index()
            PCs_proj[year, month] = (self.EOFs.xs(month, level='month') @
                                     to_proj_nearest.mul(self.wts_ss, axis=0)).T

        PCs_proj = pd.concat(PCs_proj, names=['year','month'])
        if forecast:
            PCs_proj = PCs_proj.reorder_levels(['number','year','month'])
        return PCs_proj.sort_index()
