#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as ss
from tqdm.auto import tqdm

# Custom libraries
import kdetools as kt
import scskde as sk


class Weather():
    def __init__(self, tqdm=False):
        """Process weather data for a single region-variable.

        Weather data must have been pre-processed into a standard DataFrame
        structure with unique cellIDs as columns a row index of (year, month).

        Parameters
        ----------
        tqdm : bool, optional
            Show tqdm progress bars. Defaults to False.
        """

        self.now = datetime.datetime.now()
        self.meta = {'timestamp': self.now.strftime('%Y-%m-%d %H:%M:%S')}
        self.tqdm = not tqdm

    def calc_anoms(self, data, year_range, clims):
        """Calculate anomalies from weather data for a single region-variable.

        Removing a rolling N-year climatology from the data; N=30 years centred
        moving every 5 years is the NOAA standard. Assumes that weather data is
        passed as a DataFrame with columns being unique cellIDs and a row index
        of (year, month).

        Parameters
        ----------
        data : DataFrame
            DataFrame with (year, month) MultiIndex, and cellIDs as columns.
        year_range : (int, int)
            Year range to process.
        clims : DataFrame
            DataFrame indexed by year with year_from and year_to columns,
            for calculating the climatologies for each year.
        """

        if year_range[1] > self.now.year:
            print(f'year range {year_range} cannot include the future')
            return None

        # Update metadata dict
        self.meta['anoms_clims'] = clims.to_dict('index')

        # Define the climate periods used to calculate the anomalies
        clims_map = clims.reset_index()
        clims_unique = clims_map[['year_from','year_to']].drop_duplicates()

        # Calculate climatologies from full input DataFrame
        clim_dict = {tuple(w): data.loc[slice(*w)].groupby(level='month').mean()
                     for w in clims_unique.values}
        self.clims = pd.concat({yft[0]: clim_dict[tuple(yft[1:])]
                                for yft in clims_map.values}, names=['year'])

        # Calculate 'anomalies' - deviations from local climatology
        self.anoms = (data - self.clims).dropna()

    def standardise(self, data, method='silverman', bws_dict=None, N=100,
                    buffer_bws=5, hd_wt=None, min_bw=1e-9, min_std=1e-9,
                    min_clim=-np.inf):
        """Fit 1D KDEs to input data by cell-month and transform to standard
        normally distributed z-scores

        Parameters
        ----------
        data : DataFrame
            DataFrame with (year, month) MultiIndex, and cellIDs as columns.
        method : str, optional
            Method to use for standardisation transformation. Defaults to
            'silverman', other options are 'scott', 'cv', and 'precomputed'.
            If 'precomputed' is specified, a dict of bandwidth arrays by
            month must be passed to the fit() function.
        bws_dict : dict, optional
            Dictionary of numpy arrays of KDE bandwidths. Only needed if the
            'precomputed' method is used.
        N : int, optional
            Number of points used to discretise the fitted CDF.
        buffer_bws : int, optional
            Number of bandwidths beyond data limits to extend evaluation grid.
        hd_wt : float, optional
            Weight to give Harrell-Davis quantile estimates when blending with
            kernel quantile estimates. Defaults to None (no blending). Can
            improve normalilty characteristics of transformed data if nonzero,
            but reduces extrapolation into the tail.
        min_bw : float, optional
            Minimum KDE bandwidth. Defaults to 1e-9.
        min_std : float, optional
            Cells with standard deviation less than this are filtered out.
            Defaults to 1e-9.
        min_clim : float, optional
            Minimum climatology for which to calculate standardised anomalies.
            This allows arid region-months to be excluded. Defaults to -inf.
        """

        # Update metadata dict
        self.meta['std_method'] = method
        self.meta['std_N'] = N
        self.meta['std_buffer_bws'] = buffer_bws
        self.meta['std_hd_wt'] = 0 if hd_wt is None else hd_wt
        self.meta['std_min_bw'] = min_bw
        self.meta['std_min_std'] = min_std
        self.meta['std_min_clim'] = min_clim

        self.ecdf = kt.kdecdf(N=N, buffer_bws=buffer_bws, method=method)
        self.grids, self.cdfs = {}, {}

        # Fit and transform anomalies to standard normal distributions
        Z = []
        for m, data_m in tqdm(data.groupby(level='month'), disable=self.tqdm):
            minstd_colmask = (data_m.std()>min_std)
            minclim_colmask = (self.clims.xs(m, level='month')>min_clim).any()
            cols_m = data_m.columns[minstd_colmask&minclim_colmask]
            X = data_m[cols_m]
            if method.lower() in ['silverman','scott','cv']:
                self.ecdf.fit(X, min_bw=min_bw)
            else: # 'precomputed'
                self.ecdf.fit(X, bws=bws_dict[m], min_bw=min_bw)

            # Override quantile estimates within-sample with Harrell-Davis
            if hd_wt is not None:
                hdqs = [st.mstats.hdquantiles(X.values[:,i], self.ecdf.cdfs[:-1,i])
                        for i in range(X.shape[1])]
                hdqs = np.array(np.vstack(hdqs)).T
                self.ecdf.grids[:-1,:] = hd_wt*hdqs + (1-hd_wt)*self.ecdf.grids[:-1,:]

            Z.append(pd.DataFrame(st.norm.ppf(self.ecdf.transform(X)),
                                  index=data_m.index, columns=cols_m))
            self.grids[m] = self.ecdf.grids
            self.cdfs[m] = self.ecdf.cdfs
        Z = pd.concat(Z).sort_index()

        # Zero-mean data by month to prepare for PCA
        self.Zmean = Z.groupby(level='month').mean()
        self.Z = Z - self.Zmean

        # Estimate Z-scores' normality by month using Shapiro-Wilk test p-values
        swpvals = {(m, i): st.shapiro(self.Z.xs(m, level='month')[i]).pvalue
                   for m in range(1,13) for i in self.Z.columns}
        self.swpvals_Z = pd.Series(swpvals).rename_axis(['month','geoid']).unstack('month')

    def clims_to_seasons(self, clim_year, tol=0.5, kern=[1,2,1]):
        """Identify seasons using changes in monthly climatology.

        Used for hydrological seasons, which can follow different patterns
        from the standard annual cycle of temperature outside the tropics.

        Parameters
        ----------
            clim_year : int
                Year defining the reference climatology.
            tol : float, optional
                Number in (0, 1) which multiplies the difference between the
                climatological 'ceiling' and 'floor' to calculate a threshold
                above the climatological 'floor'. Months with a climatology
                greater than the threshold are considered 'in-season'.
                Defaults to 0.5
            kern : list or ndarray, optional
                Climatology smoothing kernel with odd number of elements.
                Only non-zero elements specified - zero-padded as required.
                Defaults to [1, 2, 1].
        """

        self.meta['seas_climyear'] = clim_year
        self.meta['seas_kern'] = kern
        self.meta['seas_tol'] = tol

        # Select reference climatology
        clims_ref = self.clims.loc[clim_year]

        # Define function to do circular convolution for smoothing
        def cconv(X, kern, axis=0):
            return np.fft.irfft((np.fft.rfft(X, axis=axis).T *
                                    np.fft.rfft(kern)).T, axis=axis)

        # Smooth using centred circular convolution of monthly climatology
        kwts = np.atleast_1d(kern)/sum(kern)
        kernel = np.roll(np.pad(kwts, (0, 12-kwts.size)), -(kwts.size//2))
        clims = pd.DataFrame(cconv(clims_ref, kernel, axis=0),
                             index=clims_ref.index, columns=clims_ref.columns)

        # Make DataFrame of buffered monthly climatologies
        clims_wrap = np.vstack([clims.loc[12], clims, clims.loc[1]])

        # Calculate extrema
        extrema = np.diff(np.sign(np.diff(clims_wrap, axis=0)), axis=0)
        maxima, minima = np.where(extrema<0, 1, 0), np.where(extrema>0, 1, 0)

        # Calculate climatology threshold to identify in-season months
        ceil = pd.concat([clims.where(maxima > 0)]*3).interpolate()[12:24]
        floor = pd.concat([clims.where(minima > 0)]*3).interpolate()[12:24]
        thresh = (ceil - floor)*tol + floor
        self.ceil = ceil
        self.floor = floor
        self.thresh = thresh

        # Label seasons (in order) where climatology exceeds threshold
        mask = (clims > thresh).astype(int)
        breaks = np.clip(np.diff(np.vstack([mask.loc[[12]], mask]), axis=0),
                         0, np.inf).cumsum(axis=0)
        self.seas = pd.DataFrame(np.where(breaks>0, breaks, breaks[-1]),
                                 index=mask.index, columns=mask.columns
                                 ).where(mask>0, 0).astype(int)

    def anoms_to_PCs(self, wts=None):
        """Calculate EOFs and PCs of monthly anomalies.

        Parameters
        ----------
            wts : DataFrame, optional
                Cell-month weights to apply to anomalies with consistent index.
        """

        # Calculate weights
        if wts is None:
            self.wts = pd.DataFrame(1, index=self.Z.index, columns=self.Z.columns)
            self.meta['pca_wts'] = 'unweighted'
        else:
            self.wts = wts
            self.meta['pca_wts'] = 'weighted'

        # Keep one less EOFs/PCs than the number of unique years
        n = self.Z.index.unique(level='year').size - 1

        # Calculate EOFs and PCs for each month
        EOFs, PCs = {}, {}

        Z = self.Z * self.wts[self.Z.columns]
        for m in range(1, 13):
            X = Z.xs(m, level='month').dropna(axis=1)
            _, _, V = np.linalg.svd(X, full_matrices=False)
            EOFs[m] = pd.DataFrame(V[:n,:], columns=X.columns)
            PCs[m] = X @ EOFs[m].T

        # Convert to DataFrames
        self.EOFs = pd.concat(EOFs, names=['month','pc'])
        self.PCs = pd.concat(PCs, names=['month']).rename_axis('pc', axis=1)

        # Estimate PCs' normality by month using Shapiro-Wilk test p-values
        swpvals = {(m, pc): st.shapiro(self.PCs.xs(m, level='month')[pc]).pvalue
                   for m in range(1,13) for pc in self.PCs.columns}
        self.swpvals_PCs = pd.Series(swpvals).rename_axis(['month','pc']).unstack('month')


    def PCs_to_anoms(self, PCs, outpath=None, regvar=None, verbose=False):
        """Calculate monthly anomalies from PCs using existing EOFs.

        Parameters
        ----------
            PCs : DataFrame
                PCs of a single region-weather variable at monthly resolution.
            outpath : str, optional
                File path for output. If None, values are returned.
            regvar : str, optional
                Region-variable description for filename.
            verbose : bool, optional
                Print messages or not.

        Returns
        -------
            Zgen : DataFrame
                Generated anomalies from individual region-variables.
                Only returned if outpath is None.
        """

        writeable = outpath is not None and regvar is not None

        # Combine EOFs and PCs to get anomalies
        # If processing stochastic data, don't concat Zgen dict for efficiency
        if 'batch' in PCs.index.names:
            if outpath is None:
                print('For stochastic output, outpath must not be None')
                return None
            if verbose:
                print('Saving stochastic output to disk...')
            for b in tqdm(PCs.index.unique(level='batch'), disable=self.tqdm):
                PCs_batch = PCs.loc[b]
                Zgen = pd.concat([PC.dropna(axis=1) @ self.EOFs.loc[m]
                                  for m, PC in PCs_batch.groupby(level='month')]
                                  ).reorder_levels(PCs_batch.index.names).sort_index()
                Zgen = (Zgen/self.wts).fillna(0)
                if writeable:
                    fname = f'{regvar}_Zgen_batch{b:04}.parquet'
                    Zgen.to_parquet(os.path.join(outpath, fname))
            self.Zgen = None
        else:
            Zgen = pd.concat([PC.dropna(axis=1) @ self.EOFs.loc[m]
                              for m, PC in PCs.groupby(level='month')]
                              ).reorder_levels(PCs.index.names).sort_index()
            Zgen = (Zgen/self.wts).fillna(0)
            if writeable:
                fname = f'{regvar}_Zgen_.parquet'
                Zgen.to_parquet(os.path.join(outpath, fname))
            self.Zgen = Zgen
        return self.Zgen

    def anoms_to_data(self, Z, clim_year=None):
        """Convert anomalies back to reference data.

        Parameters
        ----------
            Z : DataFrame or dict
                Anomalies of a single region-weather variable.
            clim_year : int, optional
                Use this clim_year's reference climatology from self.clims.
                If None, use different climatologies for each year, but only
                for reconstructing historic data as-was.

        Returns
        -------
            gen : DataFrame
                Generated weather from individual region-variables.
        """

        if clim_year is None:
            clims = self.clims
        else:
            clims = self.clims.loc[clim_year]

        # Manually override grids and cdfs to do inverse transform
        ecdf = kt.kdecdf()

        # If Z is a dict, it's a stochastic batch run
        if isinstance(Z, dict):
            data = {}
            for b, Zb in Z.items():
                data_m = []
                for m, Zm in Zb.groupby(level='month'):
                    ecdf.grids = self.grids[m]
                    ecdf.cdfs = self.cdfs[m]
                    anoms_m = ecdf.inverse(ss.ndtr(Zm+self.Zmean.loc[m]))
                    data_m.append(pd.DataFrame(anoms_m, index=Zm.index,
                                                columns=Zm.columns))
                data[b] = (pd.concat(data_m).sort_index() + clims).dropna()
        else:
            data_m = []
            for m, Zm in Z.groupby(level='month'):
                ecdf.grids = self.grids[m]
                ecdf.cdfs = self.cdfs[m]
                anoms_m = ecdf.inverse(ss.ndtr(Zm+self.Zmean.loc[m]))
                data_m.append(pd.DataFrame(anoms_m, index=Zm.index,
                                            columns=Zm.columns))
            data = (pd.concat(data_m).sort_index() + clims).dropna()
        return data

    def to_file(self, outpath, desc):
        """Save model to disk.

        Model comprises climatologies, ECDF transformation data, EOFs and PCs.

        Parameters
        ----------
            outpath : str
                Output path.
            desc : str
                Description of model.
        """

        if not os.path.exists(os.path.join(outpath, desc)):
            os.makedirs(os.path.join(outpath, desc))

        # Save metadata
        with open(os.path.join(outpath, desc, 'meta.json'), 'w') as f:
            json.dump(self.meta, f, sort_keys=True, indent=4)

        self.clims.to_parquet(os.path.join(outpath, desc, 'clims.parquet'))
        self.Zmean.to_parquet(os.path.join(outpath, desc, 'Zmean.parquet'))
        self.Z.to_parquet(os.path.join(outpath, desc, 'Z.parquet'))
        self.swpvals_Z.to_parquet(os.path.join(outpath, desc, 'swpvals_Z.parquet'))
        self.EOFs.to_parquet(os.path.join(outpath, desc, 'EOFs.parquet'))
        self.PCs.to_parquet(os.path.join(outpath, desc, 'PCs.parquet'))
        self.swpvals_PCs.to_parquet(os.path.join(outpath, desc, 'swpvals_PCs.parquet'))
        if self.meta['pca_wts'] == 'weighted':
            self.wts.to_parquet(os.path.join(outpath, desc, 'PCA_wts.parquet'))

        for m in range(1, 13):
            cols = self.Z.xs(m, level='month').dropna(axis=1).columns
            grids = pd.DataFrame(self.grids[m], columns=cols)
            cdfs = pd.DataFrame(self.cdfs[m], columns=cols)
            ecdf_m = pd.concat({'grids': grids, 'cdfs': cdfs})
            ecdf_m.to_parquet(os.path.join(outpath, desc, f'ecdf_{m:02}.parquet'))

        try:
            self.seas.to_parquet(os.path.join(outpath, desc, 'seas.parquet'))
        except:
            print('No seasons identified, so not saved to file')

    def from_file(self, inpath, desc):
        """Load model from disk.

        Parameters
        ----------
            inpath : str
                Input path.
            desc : str
                Description of model.
        """

        # Load metadata
        with open(os.path.join(inpath, desc, 'meta.json'), 'r') as f:
            self.meta = json.load(f)

        self.clims = pd.read_parquet(os.path.join(inpath, desc, 'clims.parquet'))
        self.Zmean = pd.read_parquet(os.path.join(inpath, desc, 'Zmean.parquet'))
        self.Z = pd.read_parquet(os.path.join(inpath, desc, 'Z.parquet'))
        self.swpvals_Z = pd.read_parquet(os.path.join(inpath, desc, 'swpvals_Z.parquet'))
        self.EOFs = pd.read_parquet(os.path.join(inpath, desc, 'EOFs.parquet'))
        self.PCs = pd.read_parquet(os.path.join(inpath, desc, 'PCs.parquet'))
        self.swpvals_PCs = pd.read_parquet(os.path.join(inpath, desc, 'swpvals_PCs.parquet'))
        if self.meta['pca_wts'] == 'weighted':
            self.wts = pd.read_parquet(os.path.join(inpath, desc, 'PCA_wts.parquet'))

        self.ecdf = kt.kdecdf(N=self.meta['std_N'], buffer_bws=self.meta['std_buffer_bws'],
                              method=self.meta['std_method'])
        self.grids, self.cdfs = {}, {}

        for m in range(1, 13, 1):
            ecdf_m = pd.read_parquet(os.path.join(inpath, desc, f'ecdf_{m:02}.parquet'))
            self.grids[m] = ecdf_m.loc['grids'].to_numpy()
            self.cdfs[m] = ecdf_m.loc['cdfs'].to_numpy()

        try:
            self.seas = pd.read_parquet(os.path.join(inpath, desc, 'seas.parquet'))
        except:
            print('No seasons identified and saved to file')


class Model():
    def __init__(self, normalise_PCs=False, ordern=1, orderx=1,
                 bw_method='silverman_ref', bw_type='covariance', seed=42):
        """Fit and simulate from weather generator model.

        Parameters
        ----------
        normalise_PCs : boolean, optional
            Apply ECDF transform to make weather PCs more gaussian, while
            preserving their relative magnitudes. Defaults to False.
        ordern : int, optional
            Order of endogenous variables in SCSKDE model. Defaults to 1.
        orderx : int, optional
            Order of exdogenous variables in SCSKDE model. Defaults to 1.
        bw_method : str, optional
            Method for estimating KDE bandwidths. Options are 'silverman_ref'
            (default), 'silverman', 'scott', 'cv'.
        bw_type : str, optional
            Type of KDE bandwidth matrix. Options are 'covariance' (default),
            'diagonal', 'equal'.
        seed : int, optional
            Seed or random number generator state variable.
        """

        self.normalise_PCs = normalise_PCs
        self.model = sk.SCSKDE(ordern=ordern, orderx=orderx,
                               bw_method=bw_method, bw_type=bw_type)
        self.rng = np.random.RandomState(seed)
        self.now = datetime.datetime.now()
        self.meta = {'normalise_PCs': normalise_PCs,
                     'ordern': ordern, 'orderx': orderx,
                     'bw_method': bw_method, 'bw_type': bw_type, 'seed': seed,
                     'timestamp': self.now.strftime('%Y-%m-%d %H:%M:%S')}

    def load_weather_PCs(self, inpath, regvars):
        """Load historic pre-processed weather PCs from disk.

        Parameters
        ----------
        inpath : str
            Path to processed historic weather PC data.
        regvars : list or tuple
            Iterable of region-variable weather model names.
        """

        # Load weather PCs from multiple regions as a single DataFrame
        weatherPCs = {regvar: pd.read_parquet(os.path.join(inpath, regvar, 'PCs.parquet'))
                      for regvar in regvars}
        self.weatherPCs = pd.concat(weatherPCs, axis=1, names=['regvar','pc'])
        self.meta['weatherPCs'] = [os.path.join(inpath, regvar) for regvar in regvars]

    def load_tele_PCs(self, inpath, desc):
        """Load historic pre-processed SST PC data from disk.

        Parameters
        ----------
        inpath : str
            Path to processed historic SST PC data.
        desc : str
            Data description.
        """

        # Load teleconnection SST PCs
        self.telePCs = pd.read_parquet(os.path.join(inpath, desc, 'PCs.parquet'))
        self.meta['telePCs'] = os.path.join(inpath, desc)

    def load_fore_PCs(self, inpath, desc, fyear, fmonth):
        """Load forecast pre-processed SST PC data from disk.

        Parameters
        ----------
        inpath : str
            Path to processed forecast SST PC data.
        desc : str
            Data description.
        fyear : int
            Effective year of forecast.
        fmonth : int
            Effective month of forecast.
        """

        # Load forecast teleconnection (SST) PCs
        fname = f'PCs_{fyear}_{fmonth:02}.parquet'
        forePCs = pd.read_parquet(os.path.join(inpath, desc, fname))

        # Assumes ECMWF SEAS5 forecasts to set up stochastic forecast
        N_ens = 51 if fyear > 2016 else 25

        # Initial values are SST PCs from (fyear, fmonth)
        forePCs0 = pd.concat({i+1: self.telePCs.loc[[(fyear, fmonth)]]
                              for i in range(N_ens)}, names=['number'])

        self.forePCs = pd.concat([forePCs0, forePCs]).sort_index()
        self.meta['forePCs'] = os.path.join(inpath, desc, fname)


    def make_forcing(self, N_batches, year_range=(None, None), forecast=False):
        """Generate stochastic forcing by repeating historic/forecast SST PCs.

        Parameters
        ----------
            N_batches : int
                Number of batches to simulate.
            year_range : (int, int), optional
                Year range of SST PCs to repeat. Detaults to (None, None).
            forecast : bool, optional
                Flag if SST PCs are forecast. Defaults to False (historic).

        Returns
        -------
            forcing : DataFrame
                Stochastic forcing.
        """

        if forecast:
            forcing = pd.concat({b+1: self.forePCs for b in range(N_batches)},
                                 names=['batch'])
            # Merge 'batch' and 'number' levels into single 'batch' level
            years = forcing.index.get_level_values('year')
            months = forcing.index.get_level_values('month')
            forcing.index = [forcing.index.map(lambda x: (x[0]*100+x[1])
                                               ).rename('batch'), years, months]
        else:
            forcing = pd.concat({b+1: self.telePCs.loc[slice(*year_range)]
                                 for b in range(N_batches)},  names=['batch'])
        return forcing

    def PCs_to_multiPCs(self):
        """Calculate EOFs and PCs of multiple region-variables.
        """

        if self.normalise_PCs:
            self.kdecdfs = {m: kt.kdecdf(500, 5) for m in range(1, 13)}
            PCz = {}

            for m in range(1, 13):
                PCm = self.weatherPCs.xs(m, level='month')
                self.kdecdfs[m].fit(PCm)
                PCz[m] = pd.DataFrame(st.norm.ppf(self.kdecdfs[m].transform(PCm)),
                                      index=PCm.index, columns=PCm.columns)
            PCz = pd.concat(PCz, names=['month'])

            self.loc = PCz.groupby('month').mean()
            self.scale = self.weatherPCs.groupby('month').std()/PCz.groupby('month').std()
            weatherPCs = (PCz - self.loc) * self.scale
        else:
            weatherPCs = self.weatherPCs

        # Keep one less EOFs/PCs than the number of unique years
        n = weatherPCs.index.unique(level='year').size - 1

        # Calculate EOFs and PCs for each month
        multiEOFs, multiPCs = {}, {}

        for m in range(1, 13):
            X = weatherPCs.xs(m, level='month').dropna(axis=1)
            _, _, V = np.linalg.svd(X, full_matrices=False)
            multiEOFs[m] = pd.DataFrame(V[:n,:], columns=X.columns)
            multiPCs[m] = X @ multiEOFs[m].T

        # Convert to DataFrames
        self.multiEOFs = pd.concat(multiEOFs, names=['month','pc_multi'])
        self.multiPCs = pd.concat(multiPCs, names=['month','year']
                                  ).reorder_levels(['year','month']
                                                   ).sort_index(
                                                   ).rename_axis('pc_multi', axis=1)
        self.N_multiPCs = self.multiPCs.shape[1]

    def multiPCs_to_PCs(self, multiPCs, bias_correct=True, whiten=True,
                        forecast=False):
        """Calculate individual PCs from multiPCs.

        Parameters
        ----------
            multiPCs : DataFrame
                PCs of multiple region-weather variables at monthly resolution.
            bias_correct : bool, optional
                Bias correct standard deviation; mean shift handled by multiPC
                mean correction.
            whiten : bool, optional
                Decorrelate PCs using ZCA-Mahalanobis transformation by month.
            forecast : bool, optional
                Flag if forecast or not.

        Returns
        -------
            weatherPCs : DataFrame
                PCs from individual region-variables in single DataFrame.
        """

        months = multiPCs.index.unique(level='month')
        weatherPCs = pd.concat({m: multiPCs.xs(m, level='month') @
                                self.multiEOFs.xs(m, level='month')
                                for m in months}, names=['month'])
        cols = weatherPCs.columns.unique(level='regvar')

        if self.normalise_PCs:
            U = ss.ndtr(weatherPCs/self.scale + self.loc)
            weatherPCs = {}
            for m in range(1, 13):
                Um = U.xs(m, level='month')
                weatherPCs[m] = pd.DataFrame(self.kdecdfs[m].inverse(Um),
                                             index=Um.index, columns=U.columns)
            weatherPCs = pd.concat(weatherPCs, names=['month','year'])

        # Bias correct standard deviation only for non-forecast case
        if forecast:
            _, _, fmonth = multiPCs.index[0]
            if bias_correct:
                print('Warning: bias_correct=True has no effect for forecasts')

            # If whitening a forecast, do *not* whiten first month as it's fixed
            if whiten:
                weatherPCs = pd.concat([weatherPCs.xs(fmonth, level='month', drop_level=False),
                                        self.whiten(weatherPCs.drop(fmonth, level='month'))])
        else:
            # Bias correct standard deviation
            if bias_correct:
                bias_fac = (self.weatherPCs.groupby(level='month').std()/
                            weatherPCs.groupby(level='month').std())
                weatherPCs = weatherPCs * bias_fac

            if whiten:
                weatherPCs = pd.concat({c: self.whiten(weatherPCs[c]) for c in cols},
                                       names=['regvar'], axis=1)

        self.meta['bias_correct_weatherPCs'] = bias_correct
        self.meta['whiten_weatherPCs'] = whiten
        return weatherPCs.reorder_levels(multiPCs.index.names).sort_index()

    def make_depn(self, p_thresh=0.05, max_feats=3, topM=None):
        """Make endogenous dependency dictionary for SCSKDE model.

        Calculate lag-1 Spearman rank autocorrelations and do automatic
        feature selection using a 2-sided p-value threshold.

        Parameters
        ----------
            p_thresh : float, optional
                Threshold Spearman rank correlation p-value for selecting
                dependent features. Defaults to 0.05.
            max_feats : int, optional
                Maximum number of features to select. Defaults to 3.
            topM : int, optional
                Number of multiPCs to use as predictors from the previous time
                step. Defaults to the first M explaining 90% of the variance.

        Returns
        -------
            depn : dict
                Endogenous dependency dict.
        """

        if topM is None:
            varexp = self.multiPCs.groupby(level='month').var().T
            varexp = (varexp/varexp.sum())
            topM = varexp.cumsum().min(axis=1).searchsorted(0.9)

        corr, pval = {}, {}
        wcols, M = self.multiPCs.columns, self.multiPCs.shape[1]
        wcols = self.multiPCs.columns
        wcols_1 = wcols[:topM].rename(wcols.name+'-1')
        for m in range(1, 13):
            a = self.multiPCs.xs(m, level='month')
            b = self.multiPCs[wcols_1].shift(1).xs(m, level='month').dropna()
            r, p = st.spearmanr(a.reindex(b.index), b, alternative='two-sided')
            corr[m] = pd.DataFrame(r[M:,:M], columns=wcols, index=wcols_1).stack()
            pval[m] = pd.DataFrame(p[M:,:M], columns=wcols, index=wcols_1).stack()
        self.corrn = pd.concat(corr, names=['month']).unstack(['month','pc_multi'])
        self.pvaln = pd.concat(pval, names=['month']).unstack(['month','pc_multi'])
        pvaln_filt = self.pvaln.where(self.pvaln<=p_thresh)

        depn = {col: pvaln_filt[col].dropna().index.to_numpy()[:max_feats]
                if (~pvaln_filt[col].isnull()).sum()>0 else np.array([col[1]])
                for col in pvaln_filt.columns}
        self.depn = depn
        return depn

    def make_depx(self, p_thresh=0.05, max_feats=3, topM=None):
        """Make exogenous dependency dictionary for SCSKDE model.

        Calculate Spearman rank cross-correlations between teleconnection
        forcings and endogenous variables, and do automatic feature selection
        using a 2-sided p-value threshold.

        Parameters
        ----------
            p_thresh : float, optional
                Threshold Spearman rank correlation p-value for selecting
                dependent features. Defaults to 0.05.
            max_feats : int, optional
                Maximum number of features to select. Defaults to 3.
            topM : int, optional
                Number of multiPCs to use as predictors from the previous time
                step. Defaults to the first M explaining 90% of the variance.

        Returns
        -------
            depx : dict
                Exogenous dependency dict.
        """

        if topM is None:
            varexp = self.multiPCs.groupby(level='month').var().T
            varexp = (varexp/varexp.sum())
            topM = varexp.cumsum().min(axis=1).searchsorted(0.9)

        corr0, corr1, pval0, pval1 = {}, {}, {}, {}
        wcols, M = self.multiPCs.columns, self.multiPCs.shape[1]
        tcols = self.telePCs.columns[:topM].rename('tele')
        tcols_1 = self.telePCs.columns[:topM].rename('tele-1')
        for m in range(1, 13):
            a = self.multiPCs.xs(m, level='month')
            b = self.telePCs[tcols].xs(m, level='month')
            c = self.telePCs[tcols_1].shift(1).xs(m, level='month')
            ix0 = a.index.intersection(b.index)
            ix1 = a.index.intersection(c.index)
            r0, p0 = st.spearmanr(a.reindex(ix0), b.reindex(ix0), alternative='two-sided')
            r1, p1 = st.spearmanr(a.reindex(ix1), c.reindex(ix1), alternative='two-sided')
            corr0[m] = pd.DataFrame(r0[M:,:M], columns=wcols, index=tcols).stack()
            pval0[m] = pd.DataFrame(p0[M:,:M], columns=wcols, index=tcols).stack()
            corr1[m] = pd.DataFrame(r1[M:,:M], columns=wcols, index=tcols_1).stack()
            pval1[m] = pd.DataFrame(p1[M:,:M], columns=wcols, index=tcols_1).stack()

        self.corrx0 = pd.concat(corr0, names=['month']).unstack(['month','pc_multi'])
        self.pvalx0 = pd.concat(pval0, names=['month']).unstack(['month','pc_multi'])
        self.corrx1 = pd.concat(corr1, names=['month']).unstack(['month','pc_multi'])
        self.pvalx1 = pd.concat(pval1, names=['month']).unstack(['month','pc_multi'])
        self.pvalx = np.minimum(self.pvalx0, self.pvalx1)
        pvalx_filt = self.pvalx.where(self.pvalx<=p_thresh)

        depx = {col: pvalx_filt[col].dropna().index.to_numpy()[:max_feats]
                for col in pvalx_filt.columns if (~pvalx_filt[col].isnull()).sum()>0}
        self.depx = depx
        return depx

    def fit(self, p_thresh_n=0.05, max_feats_n=3, p_thresh_x=0.05, max_feats_x=3):
        """Fit SCSKDE model given historic multiPCs and teleconnection forcing.

        Parameters
        ----------
            p_thresh_n : float, optional
                Threshold Spearman rank correlation p-value for selecting
                dependent endogenous features. Defaults to 0.05.
            max_feats_n : int, optional
                Maximum number of endogenous features to select. Defaults to 3.
            p_thresh_x : float, optional
                Threshold Spearman rank correlation p-value for selecting
                dependent exogenous features. Defaults to 0.05.
            max_feats_x : int, optional
                Maximum number of exogenous features to select. Defaults to 3.
        """

        depn = self.make_depn(p_thresh_n, max_feats_n)
        depx = self.make_depx(p_thresh_x, max_feats_x)
        ix = self.multiPCs.index.intersection(self.telePCs.index)
        periods = self.multiPCs.reindex(ix).index.get_level_values('month').to_numpy()
        self.model.fit(Xn=self.multiPCs.reindex(ix).values, depn=depn,
                       Xx=self.telePCs.reindex(ix).values, depx=depx, periods=periods)
        self.meta['p_thresh_n'] = p_thresh_n
        self.meta['max_feats_n'] = max_feats_n
        self.meta['p_thresh_x'] = p_thresh_x
        self.meta['max_feats_x'] = max_feats_x

    def whiten(self, PCs):
        """Decorrelate data.

        Each month's PCs should be uncorrelated but the process of imposing
        temporal autocorrelation and dependency on teleconnection forcings
        creates a mild correlation between simulated PCs. The sum of correlated
        RVs has a greater variance than the sum of independent ones ones, so we
        remove this correlation with a ZCA-Mahalanobis transformation which is
        the optimum whitening transformation to minimise changes from the
        original data, according to Kessey et al (2016).

        Parameters
        ----------
            PCs : DataFrame
                Potentially correlated PCs.

        Returns
        -------
            PCs_df : DataFrame
                Decorrelated PCs.
        """

        # Note original MultiIndex level order
        levels = PCs.index.names

        PCs_dc = {}
        for m in PCs.index.unique(level='month'):
            X = PCs.xs(m, level='month')
            sig = np.cov(X.T)
            u, v = np.linalg.eigh(sig)
            sig_root = v * np.sqrt(np.clip(u, np.spacing(1), np.inf)) @ v.T
            W = np.linalg.inv(sig_root)
            Z = X @ W.T
            Z.columns = X.columns
            PCs_dc[m] = Z * X.std()
        return pd.concat(PCs_dc, names=['month']
                         ).reorder_levels(levels).sort_index()

    def make_multiPCs0(self, month0, N_batches):
        """Generate initial values for multiPCs for a reference month by
        sampling from the historic distribution.

        Parameters
        ----------
            month0 : int
                Reference month. Since the weather generator is a lag-1 Markov
                model, only one month is needed.
            N_batches : int
                Number of batches to simulate.

        Returns
        -------
            multiPCs0 : DataFrame
                Initial multiPCs for stochastic simulation.
        """

        sigs = self.multiPCs.xs(month0, level='month').std()
        mus = np.zeros(sigs.size)
        return pd.DataFrame(self.rng.normal(mus, sigs, size=(N_batches, sigs.size)),
                            columns=self.multiPCs.columns)

    def simulate(self, multiPCs0, forcing, seed=42, bias_correct=True,
                 whiten=True, forecast=False):
        """Generate stochastic set forced by teleconnections.

        Parameters
        ----------
            multiPCs0 : Series or DataFrame
                Initial multiPCs at single timestep as the weather generator is
                a lag-1 Markov model. If a Series of shape (n,) or a DataFrame
                of shape (n_batches, n), reshape into a (n_batches, 1, n) array.
            forcing : DataFrame
                Stochastic or forecast-stochastic teleconnection forcing.
                MultiIndex with (batch, year, month).
            seed : int, optional
                Seed or random number generator state variable.
            bias_correct : bool, optional
                Bias correct standard deviation [mean zeroed automatically].
            whiten : bool, optional
                Decorrelate PCs using ZCA-Mahalanobis transformation by month.
            forecast : bool, optional
                Flag if forecast or not.

        Returns
        -------
            stoc : DataFrame
                Stochastic multiPCs.
        """

        batches = forcing.index.unique(level='batch')
        years = forcing.index.unique(level='year')

        if forecast:
            _, _, fmonth = forcing.index[0]
            forcing_np = forcing.to_numpy().reshape(batches.size, 7, -1)
            periods_sim = (np.arange(fmonth, fmonth+7) - 1) % 12 + 1
        else:
            forcing_np = forcing.to_numpy().reshape(batches.size, years.size*12, -1)
            periods_sim = np.arange(years.size*12) % 12 + 1

        # Handle fixed or variable intial conditions
        if len(multiPCs0.shape) == 1:
            X0 = np.repeat(multiPCs0.to_numpy()[None,:], batches.size, axis=0)[:,None]
        else:
            X0 = multiPCs0.to_numpy()[:,None]

        # Run simulation and post-process
        if forecast:
            stoc_raw = self.model.simulate(7, X0=X0, Xx=forcing_np,
                                           batches=batches.size,
                                           periods=periods_sim, seed=seed)
        else:
            stoc_raw = self.model.simulate(years.size*12, X0=X0, Xx=forcing_np,
                                           batches=batches.size,
                                           periods=periods_sim, seed=seed)
        stoc = pd.DataFrame(stoc_raw.reshape(-1, self.N_multiPCs),
                            index=forcing.index).rename_axis('pc_multi', axis=1)

        # Don't zero-mean and bias correct simulated multiPCs for forecasts
        if forecast:
            if bias_correct:
                print('Warning: bias_correct=True has no effect for forecasts')

            # If whitening a forecast, do *not* whiten first month as it's fixed
            if whiten:
                stoc = pd.concat([stoc.xs(fmonth, level='month', drop_level=False),
                                  self.whiten(stoc.drop(fmonth, level='month'))])
        else:
            stoc_zm = stoc - stoc.groupby(level='month').mean()
            if bias_correct:
                bias_fac = (self.multiPCs.groupby(level='month').std()/
                            stoc_zm.groupby(level='month').std())
                stoc = stoc_zm * bias_fac
            else:
                stoc = stoc_zm

            if whiten:
                stoc = self.whiten(stoc)

        self.meta['seed'] = seed
        self.meta['N_batches'] = len(batches)
        self.meta['year_range'] = (years.min(), years.max())
        self.meta['bias_correct_sim'] = bias_correct
        self.meta['whiten_sim'] = whiten
        return stoc.sort_index()

    def to_file(self, outpath, desc):
        """Save model to disk.

        Parameters
        ----------
            outpath : str
                Output path.
            desc : str
                Description of model.
        """

        # TODO Create model.from_file()
        return None

    def from_file(self, inpath, desc):
        """Load model from disk.

        Parameters
        ----------
            inpath : str
                Input path.
            desc : str
                Description of model.
        """

        # TODO Create model.from_file()
        return None
