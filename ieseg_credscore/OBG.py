import numpy as np
import pandas as pd

class OBGEncoder:
    def __init__(self, pred_var, target_var):
        self.pred_var = pred_var
        self.target_var = target_var
        self.cont = False

    def _step_OBG(self, var, target, max_delta, min_bins):
        
        # convert variable to string
        var = var.astype(str).copy()

        # check stopping criteria: min bins
        if len(np.unique(var)) <= min_bins:
            return var

        # compute odds
        ct = pd.crosstab(var, target)
        odds = np.array(ct.iloc[:,1] / (ct.iloc[:,0] + ct.iloc[:,1]))
        diff = np.abs(np.triu(np.subtract.outer(odds, odds)))
        diff[diff==0] = np.inf

        # check stopping criteria: max_delta
        if np.min(diff) > max_delta:
            return var

        # get index for levels with min delta
        a, b = np.unravel_index(diff.argmin(), diff.shape)
        
        name_a, name_b = list(ct.index[[a,b]])
        name_update = f"{name_a}_{name_b}"

        # group levels of variable
        var_update = var.replace(name_a, name_update).replace(name_b, name_update)

        # recursive call
        return self._step_OBG(var_update, target, max_delta, min_bins)
    
    # convert continous variable to categorical
    def cont2cat(self, var):
        return pd.qcut(var, q=self.q, labels=False, duplicates='drop').fillna(-1).astype(int)

    # lookup quantiles from train set
    def _lookup_cat(self, x):
        if np.isnan(x): return "nan"
        else:
            for key, val in self.lookup.items():
                if (x <= val["orig_max"]) & (x >= val["orig_min"]):
                    return str(key)
            
            # if value cannot be matched
            return "nan"

    # fit on training data
    def fit(self, df, max_delta=0.05, min_bins=3, **kwargs):
        
        # detect data type
        var_type = df[self.pred_var].dtype

        if var_type in [float, int]:
            self.cont = True
            self.q = kwargs.get("q", 10)
            var_cat = self.cont2cat(df[self.pred_var])
            concat = pd.concat([df[self.pred_var], var_cat], axis=1)
            concat.columns = ["orig", "bin"]

            bin_range = concat.groupby('bin').agg({'orig':['min', 'max']}).dropna()
            bin_range.columns = ['_'.join(col).strip() for col in bin_range.columns.values]
            self.lookup = bin_range.to_dict(orient="index")

            print(f"Detected '{var_type}' variable, automatically transformed using q={self.q} quantiles ... ")

            res = self._step_OBG(var_cat, df[self.target_var], max_delta, min_bins)

        elif var_type in [object]:
            res = self._step_OBG(df[self.pred_var], df[self.target_var], max_delta, min_bins)
        
        else:
            raise Exception(f"Unknown data type '{var_type}'")
        
        self.fit_dict = {}
        groups = [i.split("_") for i in np.unique(res)]

        for i, group in enumerate(groups):
            if isinstance(group, list):
                for level in group:
                    self.fit_dict[level] = f"{self.pred_var}_{i+1}"
        
        if ("nan" not in self.fit_dict.keys()) & ("-1" in self.fit_dict.keys()):
            self.fit_dict["nan"] = self.fit_dict["-1"]
       
        return
    
    # fit and transform training data
    def fit_transform(self, df, max_delta=0.05, min_bins=3, **kwargs):
        self.fit(df, max_delta, min_bins, **kwargs)
        return self.transform(df)
    
    # transform 
    def transform(self, df, impute=False, impute_value=""):
        df_copy = df.copy()
        transform_vec = np.vectorize(lambda level: self.fit_dict[str(level)])

        if self.cont:
            lookup_func = np.vectorize(self._lookup_cat)
            df_copy[self.pred_var] = lookup_func(df_copy[self.pred_var])

        if impute:
            if ("Missing" in self.fit_dict.keys()): impute_value = "Missing"
            elif ("nan" in self.fit_dict.keys()): impute_value = "nan"
            else: raise Exception("No impute value found. Please specify impute_value!")

            unseen = set(df_copy[self.pred_var].astype(str)) - set(self.fit_dict.keys())
            if (len(unseen) > 0):
                df_copy.loc[df_copy[self.pred_var].isin(unseen), self.pred_var] = impute_value
                print(f"Imputed {len(unseen)} values...")

        pred_bin = transform_vec(df_copy[self.pred_var])
        df_copy[self.pred_var] = pred_bin
        return df_copy