#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as ss
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# Custom libraries
import quadgrid as qg
import climperiods
import kdetools as kt
import scskde as sk

from .telesst import TeleSST


class Weather():
    def __init__(self, tqdm=True):
        """Process weather data for a single region-variable.

        Weather data must have been pre-processed into a standard structure,
        with columns being unique cellIDs and a row index of (year, month).
        """

        self.tqdm = tqdm
        self.now = datetime.datetime.now()
        self.EOFs = None
        self.PCs = None
        self.varexp = None

    def calc_anoms(self, weather_data, year_range, N_KDE=100, buffer_bws=3):
        """Calculate anomalies from weather data for a single region-variable.

        Data is first detrended by removing a rolling N-year climatology
        (N = 30 years centred is the NOAA standard) then it is converted to
        standardised anomalies by fitting a KDE to each cell-month. Assumes
        that weather data is saved in parquet format with columns being unique
        cellIDs and a row index of (year, month).

        Parameters
        ----------
        weather_data : DataFrame
            DataFrame with (year, month) MultiIndex, and qid (quadtreeID) columns.
        year_range : (int, int)
            Year range to process.
        N_KDE : int, optional
            Number of points used to discretise the KDE.
        buffer_bws : int, optional
            Number of bandwidths beyond data limits to extend evaluation
            range over, to handle extrapolation.
        """

        # Load the raw weather data and pivot to cell-months
        data = weather_data.unstack('month')
        self.cols_all = data.columns

        #  Filter out cell-months if IQR == 0, i.e. no interannual variability
        iqrs = np.diff(np.quantile(data, [0.25, 0.75], axis=0), axis=0)[0]
        self.cols = data.columns[iqrs>0]
        self.data = data[self.cols]

        if year_range[1] > self.now.year:
            print(f'year range {year_range} cannot include the future')
            return None

        # Define the climate periods used to calculate the anomalies
        clims_map = climperiods.clims(*year_range)
        clims_unique = clims_map.drop_duplicates()

        # Calculate climatologies from full input DataFrame, anomalies from valid
        clims = {window: data.loc[slice(*window),:].mean()
                 for window in clims_unique.itertuples(index=False, name=None)}
        self.clims = pd.concat({yft[0]: clims[yft[1:]]
                                for yft in clims_map.itertuples(name=None)},
                                names=['year'], axis=1).T
        anoms = self.data.reindex(self.clims.index) - self.clims[self.cols]

        # Fit 1D KDEs to each cell-month detrended anomalies and transform
        self.ecdf = kt.kdecdf(N=N_KDE, buffer_bws=buffer_bws)
        self.ecdf.fit(anoms)
        anoms_tr = pd.DataFrame(st.norm.ppf(self.ecdf.transform(anoms)),
                                index=anoms.index, columns=anoms.columns)

        # Zero-mean data by month prior to PCA
        self.anoms_tr_mean = anoms_tr.mean()
        self.zin = anoms_tr - self.anoms_tr_mean

    def clims_to_seasons(self, clim_year, buffer=(1, 1), max_nseas=2):
        """Identify seasons using changes in monthly climatology.

        Used for hydrological seasons, which can follow different patterns
        from the standard annual cycle of temperature outside the tropics.

        Parameters
        ----------
            clim_year : int
                Reference climatology to use.
            buffer : (int, int), optional
                Number of months to buffer either side of seasonal max or min.
                Defaults to (1, 1).
            max_nseas : int, optional
                Maximum number of seasons to apply buffer. Defaults to 2.
                For cells with more nominal seasons as identified by local
                peaks, buffer is only applied to the largest max_nseas ones.
        """

        # Make DataFrame of buffered monthly climatologies
        clims = self.clims.loc[clim_year].unstack('month').T.sort_index()
        clims_wrap = np.vstack([clims.loc[12], clims, clims.loc[1]])

        # Annual 25th and 75th climate percentiles - used heuristically for
        # more robust identification of peaks
        cq25, cq75 = clims.rank(axis=0)>3, clims.rank(axis=0)<9

        # Calculate extrema
        extrema = np.diff(np.sign(np.diff(clims_wrap, axis=0)), axis=0)
        maxima, minima = np.where(extrema<0, 1, 0), np.where(extrema>0, 1, 0)
        maxima_filt = (maxima>0) & cq25
        minima_filt = (minima>0) & cq75
        peaks = np.where(maxima_filt, np.cumsum(maxima_filt, axis=0), 0)
        troughs = np.where(minima_filt, -np.cumsum(minima_filt, axis=0), 0)

        # Buffer months around extrema only if # seasons <= max_nseas
        self.nseas = pd.Series(np.count_nonzero(peaks, axis=0),
                               index=clims.columns)
        sflag = np.where(self.nseas<=max_nseas, 1, 0)
        buff_rng = range(-buffer[0], buffer[1]+1)
        peaks_buff = np.sum([np.roll(peaks, i, axis=0) * sflag
                             for i in buff_rng if i!=0], axis=0)
        troughs_buff = np.sum([np.roll(troughs, i, axis=0) * sflag
                               for i in buff_rng if i!=0], axis=0)
        self.maxima = peaks
        self.minima = troughs
        self.seas_maxima = pd.DataFrame(peaks + peaks_buff, index=clims.index,
                                        columns=clims.columns)
        self.seas_minima = pd.DataFrame(troughs + troughs_buff, index=clims.index,
                                        columns=clims.columns)

    def anoms_to_PCs(self, wts=None):
        """Calculate EOFs and PCs of monthly standard anomalies.

        Parameters
        ----------
            wts : Series, optional
                Weights to apply to standard anomalies with index consistent
                with the columns of the already-computed standard anomalies.
        """

        # Calculate weights
        if wts is None:
            self.wts = 1

        # Calculate EOFs and PCs for each month
        EOFs, PCs, varexp = {}, {}, {}
        pca = PCA()
        for m in tqdm(range(1, 13), disable=self.tqdm):
            X = self.zin.xs(m, level='month', axis=1) * self.wts
            pca.fit(X)
            EOFs[m] = pd.DataFrame(pca.components_, columns=X.columns)
            PCs[m] = X @ EOFs[m].T
            varexp[m] = pca.explained_variance_ratio_

            # Align EOFs of successive months for ease of interpretation
            if m > 1:
                cols = EOFs[m].columns.intersection(EOFs[m-1].columns)
                sgn = np.sign(np.diag(EOFs[m][cols] @ EOFs[m-1][cols].T))
                EOFs[m] = EOFs[m] * sgn[:,None]
                PCs[m] = PCs[m] * sgn[None,:]

        # Convert to DataFrames
        self.EOFs = pd.concat(EOFs, names=['month','qid'], axis=1).rename_axis('pc')
        self.PCs = pd.concat(PCs, names=['month']).rename_axis('pc', axis=1)
        self.varexp = pd.DataFrame(varexp).rename_axis('month', axis=1
                                                       ).rename_axis('pc', axis=0)

    def PCs_to_anoms(self, PCs, outpath=None, regvar=None):
        """Calculate monthly standard anomalies from PCs using existing EOFs.

        Parameters
        ----------
            PCs : DataFrame
                PCs of a single region-weather variable at monthly resolution.
            outpath : str, optional
                File path to write output. If None, values are return
            regvar : str, optional
                Region-variable name for writing to file.

        Returns
        -------
            zgen : DataFrame
                Generated standard anomalies from individual region-variables.
                Only returned if outpath is None.
        """

        # Combine EOFs and PCs and add back mean of transformed anomalies
        months = PCs.index.unique(level='month')
        if outpath is None:
            self.zgen = pd.concat({m: PCs.xs(m, level='month') @ self.EOFs.xs(m, level='month', axis=1)
                                   for m in tqdm(months, disable=self.tqdm)}, names=['month'], axis=1
                                   ).reorder_levels(['qid','month'], axis=1
                                                    ).reindex(self.cols, axis=1
                                                              ).add(self.anoms_tr_mean)
            return self.zgen
        elif regvar is not None:
            # Stochastic or forecast-stochastic run
            if 'batch' in PCs.index.names:
                for batch in PCs.index.unique(level='batch'):
                    PCs_batch = PCs.loc[batch]
                    zgen = pd.concat({m: PCs_batch.xs(m, level='month') @ self.EOFs.xs(m, level='month', axis=1)
                                      for m in tqdm(months, disable=self.tqdm)}, names=['month'], axis=1
                                      ).reorder_levels(['qid','month'], axis=1
                                                       ).reindex(self.cols, axis=1
                                                                 ).add(self.anoms_tr_mean).stack('month')
                    zgen.to_parquet(os.path.join(outpath, f'{regvar}_zgen_batch{batch:03}.parquet'))
        else:
            print('Both or neither outpath and regvar should be None.')

    def anoms_to_data(self, z, clim_year=None):
        """Convert standard anomalies back to reference data using inverse
         standardisation transformations.

        Parameters
        ----------
            z : DataFrame
                Standard anomalies of a single region-weather variable
                at monthly resolution.
            clim_year : int, optional
                Year from self.clims to use as reference climatology.
                Corresponds to a pre-defined reference period. If None,
                use different climatologies for each year, but this is
                only applicable for reconstructing historic data as-was.

        Returns
        -------
            gen : DataFrame
                Generated weather from individual region-variables.
        """
        # Invert the transform
        anoms = pd.DataFrame(self.ecdf.inverse(np.clip(ss.ndtr(z),
                                               a_min=self.ecdf.cdfs.min(axis=0),
                                               a_max=self.ecdf.cdfs.max(axis=0))),
                                               index=z.index, columns=z.columns)
        if clim_year is None:
            return anoms + self.clims
        else:
            return anoms + self.clims.loc[clim_year]

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
        self.cols.to_frame().to_csv(os.path.join(outpath, desc, 'columns.csv'),
                                    index=False)
        self.clims.to_parquet(os.path.join(outpath, desc, 'clims.parquet'))
        self.ecdf.to_file(os.path.join(outpath, desc), 'ecdf')
        self.anoms_tr_mean.unstack('month').to_parquet(os.path.join(outpath, desc, 'anoms-tr-mean.parquet'))
        self.EOFs.to_parquet(os.path.join(outpath, desc, 'EOFs.parquet'))
        self.PCs.to_parquet(os.path.join(outpath, desc, 'PCs.parquet'))

    def from_file(self, inpath, desc):
        """Load model from disk.

        Parameters
        ----------
            inpath : str
                Input path.
            desc : str
                Description of model.
        """

        cols_df = pd.read_csv(os.path.join(inpath, desc, 'columns.csv'))
        self.cols = pd.MultiIndex.from_frame(cols_df)
        self.clims = pd.read_parquet(os.path.join(inpath, desc, 'clims.parquet'))
        self.ecdf = kt.kdecdf()
        self.ecdf.from_file(os.path.join(inpath, desc), 'ecdf')
        self.anoms_tr_mean = pd.read_parquet(os.path.join(inpath, desc, 'anoms-tr-mean.parquet')).stack('month')
        self.EOFs = pd.read_parquet(os.path.join(inpath, desc, 'EOFs.parquet'))
        self.PCs = pd.read_parquet(os.path.join(inpath, desc, 'PCs.parquet'))

    def to_oasis(self, haz_df, outpath, kind='stoc', bounds=(0, 0, 0.5, 1),
                 nbins=51, qid_res=5/60):
        """Convert data to Oasis hazard format.

        This method is consistent with Oasis ODS v{x.xx} and produces 4 tables:
        - event : list of event_id values
        - areaperil_dict : mapping between areaperil_id and (lon, lat) coords
        - intensity_bin_dict : mapping from intensity_bin_index to hazard
        - footprint : list of event_id, areaperil_id and intensity_bin_index
        - occurrence_lt : event occurrence lookup table

        The method needs additional information to decide how to encode the
        raw hazard data; this information is defined in the method arguments.
        """

        # event and occurence_lt
        occlt = haz_df.index.to_frame(index=False)
        if kind in ['hist','hist-detr']:
            event = pd.Series([f'{y:04}{m:02}' for y, m in occlt.values],
                              name='event_id').astype(int)
            occurrence_lt = occlt.add_prefix('occ_')
        elif kind == 'stoc':
            event = pd.Series([f'{b}{y:04}{m:02}' for b, y, m in occlt.values],
                              name='event_id').astype(int)
            occurrence_lt = occlt.drop('batch', axis=1).add_prefix('occ_')
        elif kind == 'stoc-fore':
            event = pd.Series([f'{b}{n:02}{y:04}{m:02}' for b, n, y, m in occlt.values],
                              name='event_id').astype(int)
            occurrence_lt = occlt.drop(['batch','number'], axis=1).add_prefix('occ_')
        else:
            print('kind must be one of hist, hist-detr, stoc, stoc-fore')
            return None

        occurrence_lt['occ_day'] = 1
        occurrence_lt['event_id'] = event
        occurrence_lt['period_no'] = occurrence_lt['occ_year']
        occurrence_lt = occurrence_lt[['event_id','period_no','occ_year','occ_month','occ_day']]

        # areaperil_dict
        lons, lats = qg.qids2lls(haz_df.columns.astype(int), res_target=qid_res)
        areaperil_dict = pd.DataFrame({'areaperil_id': haz_df.columns.values,
                                       'lat': lats, 'lon': lons})

        # intensity_bin_dict
        llb, lb, ub, uub = bounds
        hazbins = np.unique(np.r_[llb, np.linspace(lb, ub, nbins), uub])
        bin_ix = range(1, hazbins.size)
        intensity_bin_dict = pd.DataFrame(zip(bin_ix, hazbins[:-1], hazbins[1:]),
                                          columns=['bin_index','bin_from','bin_to'])

        # footprint
        footprint = haz_df.rename_axis('areaperil_id', axis=1)
        footprint.index = pd.Index(event)
        footprint = pd.cut(footprint.stack(), bins=hazbins, labels=bin_ix,
                           right=False).astype(int).rename('intensity_bin_index').reset_index()
        footprint['prob'] = 1.

        # Write files to disk [return for testing]
        return {'event': event, 'areaperil': areaperil_dict,
                'intensity_bin': intensity_bin_dict, 'footprint': footprint,
                'occurrence_lt': occurrence_lt}

class Model():
    def __init__(self, weatherpc_inpath, telehist_inpath, telefore_inpath=None):
        """Fit and apply weather generation model.

        Parameters
        ----------
        weatherpc_inpath : str
            Path to processed historic weather data.
        telehist_inpath : str
            Path to processed historic teleconnection data.
        telefore_inpath : str
            Path to processed historic teleconnection data.
        """

        self.weatherpc_inpath = weatherpc_inpath
        self.telehist_inpath = telehist_inpath
        self.telefore_inpath = telefore_inpath

    def load_weather_data(self, names):
        """Load historic pre-processed weather data from disk.

        Parameters
        ----------
        names : list or tuple
            Iterable of region-level weather model names.
        """

        # Load weather PCs from multiple regions as a single DataFrame
        weatherPCs = {name: pd.read_parquet(os.path.join(self.weatherpc_inpath, name, 'PCs.parquet'))
                      for name in names}
        self.weatherPCs = pd.concat(weatherPCs, axis=1, names=['regvar','pc'])

    def load_tele_data(self, name):
        """Load historic pre-processed teleconnection data from disk.

        Parameters
        ----------
        name : str
            Name of teleconnection data model.
        """

        # Load teleconnection SST PCs
        self.tele = TeleSST()
        self.tele.from_file(self.telehist_inpath, name)

    def to_multiPCs(self):
        """Calculate EOFs and PCs of multiple region-variables.
        """

        # Calculate EOFs and PCs for each month
        multiEOFs, multiPCs, varexp = {}, {}, {}
        pca = PCA()
        for m in range(1, 13):
            X = self.weatherPCs.xs(m, level='month')
            pca.fit(X)
            multiEOFs[m] = pd.DataFrame(pca.components_, columns=X.columns)
            multiPCs[m] = X @ multiEOFs[m].T
            varexp[m] = pca.explained_variance_ratio_

            # Align EOFs of successive months for ease of interpretation
            if m > 1:
                cols = multiEOFs[m].columns.intersection(multiEOFs[m-1].columns)
                sgn = np.sign(np.diag(multiEOFs[m][cols] @ multiEOFs[m-1][cols].T))
                multiEOFs[m] = multiEOFs[m] * sgn[:,None]
                multiPCs[m] = multiPCs[m] * sgn[None,:]

        # Convert to DataFrames
        self.multiEOFs = pd.concat(multiEOFs, names=['month','pc_multi'])
        self.multiPCs = pd.concat(multiPCs, names=['month','year']
                                  ).reorder_levels(['year','month']
                                                   ).sort_index(
                                                   ).rename_axis('pc_multi', axis=1)
        self.varexp = pd.DataFrame(varexp).rename_axis('month', axis=1
                                                       ).rename_axis('pc_multi', axis=0)
        self.N_multiPCs = self.multiPCs.shape[1]

    def from_multiPCs(self, multiPCs, bias_correct=True, whiten=True):
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

        Returns
        -------
            weatherPCs : DataFrame
                PCs from individual region-variables in single DataFrame.
        """

        months = multiPCs.index.unique(level='month')
        weatherPCs = pd.concat({m: multiPCs.xs(m, level='month') @
                                self.multiEOFs.xs(m, level='month')
                                for m in months}, names=['month'])

        # Bias correct standard deviation
        if bias_correct:
            bias_fac = self.weatherPCs.groupby(level='month').std()/weatherPCs.groupby(level='month').std()
            weatherPCs = weatherPCs * bias_fac

        if whiten:
            cols = weatherPCs.columns.unique(level='regvar')
            return pd.concat({c: self.whiten(weatherPCs[c]) for c in cols},
                             names=['regvar'], axis=1)
        else:
            return weatherPCs

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
                step. Defaults to the first M explaining 99% of the variance.

        Returns
        -------
            depn : dict
                Endogenous dependency dict.
        """

        if topM is None:
            topM = self.varexp.cumsum().min(axis=1).searchsorted(0.99)

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
                step. Defaults to the first M explaining 99% of the variance.

        Returns
        -------
            depx : dict
                Exogenous dependency dict.
        """

        if topM is None:
            topM = self.tele.varexp.cumsum().min(axis=1).searchsorted(0.99)

        corr0, corr1, pval0, pval1 = {}, {}, {}, {}
        wcols, M = self.multiPCs.columns, self.multiPCs.shape[1]
        tcols = self.tele.PCs.columns[:topM].rename('tele')
        tcols_1 = self.tele.PCs.columns[:topM].rename('tele-1')
        for m in range(1, 13):
            a = self.multiPCs.xs(m, level='month')
            b = self.tele.PCs[tcols].xs(m, level='month')
            c = self.tele.PCs[tcols_1].shift(1).xs(m, level='month')
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
        ix = self.multiPCs.index.intersection(self.tele.PCs.index)
        periods = self.multiPCs.reindex(ix).index.get_level_values('month').to_numpy()
        self.model = sk.SCSKDE(ordern=1, orderx=1, bw_method='silverman_ref')
        self.model.fit(Xn=self.multiPCs.reindex(ix).values, depn=depn,
                       Xx=self.tele.PCs.reindex(ix).values, depx=depx, periods=periods)

    def make_stoc_tele(self, year_range, N_batches):
        """Generate stochastic teleconnections by repeating historic SST PCs.

        Parameters
        ----------
            year_range : (int, int)
                Year range of SST PCs to repeat.
            N_batches : int
                Number of batches to simulate.

        Returns
        -------
            stoc_tele : DataFrame
                Stochastic teleconnections.
        """

        stoc_tele = pd.concat({b+1: self.tele.PCs.loc[slice(*year_range)]
                               for b in range(N_batches)},
                               names=['batch','year','month'])
        return stoc_tele

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
        PCs_dc = pd.concat(PCs_dc, names=['month'])
        levels = PCs_dc.index.names
        PCs_dc = PCs_dc.reorder_levels(levels[1:]+[levels[0]]).sort_index()
        return PCs_dc

    def make_multiPCs0(self, month0, N_batches, seed=42):
        """Generate initial values for multiPCs for a reference month by
        sampling from the historic distribution.

        Parameters
        ----------
            month0 : int
                Reference month. Since the weather generator is a lag-1 Markov
                model, only one month is needed.
            N_batches : int
                Number of batches to simulate.
            seed : int, optional
                Seed or random number generator state variable.

        Returns
        -------
            multiPCs0 : DataFrame
                Initial multiPCs for stochastic simulation.
        """

        rng = np.random.RandomState(seed)
        sigs = self.multiPCs.xs(month0, level='month').std()
        mus = np.zeros(sigs.size)
        return pd.DataFrame(rng.normal(mus, sigs, size=(N_batches, sigs.size)),
                            columns=self.multiPCs.columns)

    def simulate(self, multiPCs0, telePCs, seed=42, bias_correct=True, whiten=True):
        """Generate stochastic set forced by teleconnections.

        Parameters
        ----------
            multiPCs0 : Series or DataFrame
                Initial multiPCs at single timestep as the weather generator is
                a lag-1 Markov model. If a Series of shape (n,) or a DataFrame
                of shape (n_batches, n), reshape into a (n_batches, 1, n) array.
            telePCs : DataFrame
                Teleconnection forcing.
            seed : int, optional
                Seed or random number generator state variable.
            bias_correct : bool, optional
                Bias correct standard deviation [mean zeroed automatically].
            whiten : bool, optional
                Decorrelate PCs using ZCA-Mahalanobis transformation by month.

        Returns
        -------
            stoc : DataFrame
                Stochastic multiPCs.
        """

        batches = telePCs.index.unique(level='batch')
        years = telePCs.index.unique(level='year')
        telePCs_np = telePCs.to_numpy().reshape(batches.size, years.size*12, -1)

        # Define period (i.e. month) labels
        periods_sim = np.arange(years.size*12) % 12 + 1

        # Handle fixed or variable intial conditions
        if len(multiPCs0.shape) == 1:
            X0 = np.repeat(multiPCs0.to_numpy()[None,:], batches.size, axis=0)[:,None]
        else:
            X0 = multiPCs0.to_numpy()[:,None]

        # Run simulation and post-process
        stoc_raw = self.model.simulate(years.size*12, X0=X0, Xx=telePCs_np,
                                       batches=batches.size,
                                       periods=periods_sim, seed=seed)
        stoc_df = pd.DataFrame(stoc_raw.reshape(-1, self.N_multiPCs),
                               index=telePCs.index).rename_axis('pc_multi', axis=1)

        # Zero-mean simulated PCs and bias correct standard deviation
        stoc_zm = stoc_df - stoc_df.groupby(level='month').mean()
        if bias_correct:
            bias_fac = self.multiPCs.groupby(level='month').std()/stoc_zm.groupby(level='month').std()
            stoc = stoc_zm * bias_fac
        else:
            stoc = stoc_zm

        if whiten:
            return self.whiten(stoc)
        else:
            return stoc

    def forecast(self, fmonth, fyear, N_batches, seed=42):
        """Generate forecast-stochastic set forced by SEAS5 SST forecast.

        Parameters
        ----------
            fmonth : int
                Effective month of forecast.
            fyear : int
                Effective year of forecast.
            N_batches : int
                Number of batches to simulate.
            seed : int, optional
                Seed or random number generator state variable.

        Returns
        -------
            stoc_forecast : DataFrame
                Stochastic forecast multiPCs.
        """

        # Assumes ECMWF SEAS5 forecasts to set up stochastic forecast
        N_ens = 51 if fyear > 2016 else 25

        # Define initial values
        multiPCs_init = np.repeat(self.multiPCs.loc[fyear, fmonth].values[None,:],
                                  N_ens*N_batches, axis=0)[:,None]
        telePCs_init = pd.concat({i+1: self.tele.PCs.loc[[(fyear, fmonth)]]
                                  for i in range(N_ens)}, names=['number'])

        # Load forecast SST data, project onto SST EOFs and convert to numpy
        self.tele.calc_anoms_forecast(self.telefore_inpath, fyear, fmonth)
        self.tele.anoms_fore = self.tele.anoms_fore.assign_coords(number=self.tele.anoms_fore.number+1)
        telePCs_fore = self.tele.project(self.tele.anoms_fore)
        telePCs = pd.concat([telePCs_init, telePCs_fore]).sort_index()
        telePCs = pd.concat({batch+1: telePCs
                             for batch in range(N_batches)}, names=['batch'])
        telePCs_np = telePCs.to_numpy().reshape(N_ens*N_batches, 7, -1)

        # Define period (i.e. month) labels
        periods_sim = (np.arange(fmonth, fmonth+7) - 1) % 12 + 1

        # Run simulation and post-process
        stoc_forecast = self.model.simulate(7, X0=multiPCs_init, Xx=telePCs_np,
                                            batches=N_ens*N_batches,
                                            periods=periods_sim, seed=seed)
        return pd.DataFrame(stoc_forecast.reshape(-1, self.N_multiPCs),
                            index=telePCs.index).rename_axis('pc_multi', axis=1)

    def to_file(self, outpath, desc):
        """Save model to disk.

        Parameters
        ----------
            outpath : str
                Output path.
            desc : str
                Description of model.
        """

        if not os.path.exists(os.path.join(outpath, desc)):
            os.makedirs(os.path.join(outpath, desc))
        self.cols.to_frame().to_csv(os.path.join(outpath, desc, 'columns.csv'),
                                    index=False)
        self.clims.to_parquet(os.path.join(outpath, desc, 'clims.parquet'))
        self.ecdf.to_file(os.path.join(outpath, desc), 'ecdf')
        self.anoms_tr_mean.unstack('month').to_parquet(os.path.join(outpath, desc, 'anoms-tr-mean.parquet'))
        self.EOFs.to_parquet(os.path.join(outpath, desc, 'EOFs.parquet'))
        self.PCs.to_parquet(os.path.join(outpath, desc, 'PCs.parquet'))

    def from_file(self, inpath, desc):
        """Load model from disk.

        Parameters
        ----------
            inpath : str
                Input path.
            desc : str
                Description of model.
        """

        cols_df = pd.read_csv(os.path.join(inpath, desc, 'columns.csv'))
        self.cols = pd.MultiIndex.from_frame(cols_df)
        self.clims = pd.read_parquet(os.path.join(inpath, desc, 'clims.parquet'))
        self.ecdf = kt.kdecdf()
        self.ecdf.from_file(os.path.join(inpath, desc), 'ecdf')
        self.anoms_tr_mean = pd.read_parquet(os.path.join(inpath, desc, 'anoms-tr-mean.parquet')).stack('month')
        self.EOFs = pd.read_parquet(os.path.join(inpath, desc, 'EOFs.parquet'))
        self.PCs = pd.read_parquet(os.path.join(inpath, desc, 'PCs.parquet'))
