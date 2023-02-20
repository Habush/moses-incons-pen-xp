import jax
import jax.numpy as jnp
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def generate_synthetic_data(*, key, num_tf, num_genes,
                            tf_on, num_samples, binary, val_tf=4):

    p = num_tf + (num_tf * num_genes)

    keys = jax.random.split(key, num_tf)

    def generate_tfs(key):
        tf = jax.random.normal(key=key, shape=(num_samples, ))
        return tf

    def generate_genes(key, tf):
        key_rmh = jax.random.split(key, num_genes)

        def generate_single_gene(i, key):
            gene = tf + 0.51*jax.random.normal(key=key, shape=(num_samples,))
            return i+1, gene

        _, genes = jax.lax.scan(generate_single_gene, 0, key_rmh)

        return genes

    tfs = jax.vmap(generate_tfs)(keys)
    genes = jax.vmap(generate_genes)(keys, tfs)

    key_tf, key_genes = jax.random.split(key, 2)

    idx_on = jax.random.choice(key_tf, jnp.arange(num_tf), shape=(tf_on, ), replace=False)

    betas = jnp.zeros(p)

    X = jnp.zeros((num_samples, p))


    val_tf = val_tf
    val_gene = val_tf/np.sqrt(10)

    k = num_genes + 1

    for i in range(p):
        X = X.at[:,i].set(tfs[i])
        for j in range(i+1, i+k):
            X = X.at[:,j].set(genes[i, j])

    # num_pos_reg = int(num_genes*perc_pos)
    # if perc_pos < 1:
    #     pos_reg_idx = jax.random.choice(key_genes, jnp.arange(num_genes), shape=(num_pos_reg, ))
    # else:


    for i in range(tf_on):
        idx = idx_on[i]*k
        betas = betas.at[idx].set(val_tf)
        for j in range(idx+1, idx+k):
            # if j in pos_reg_idx: # positively regulated gene
            #     betas = betas.at[j].set(val_gene)
            #
            # else: # negatively regulated gene
            #     betas = betas.at[j].set(-val_gene)
            betas = betas.at[j].set(val_gene)



    y = jnp.dot(X, betas)

    if binary: # return classification data
        p = jax.nn.sigmoid(y)
        y = (jax.vmap(jax.random.bernoulli, in_axes=(None, 0))(key, p))*1.
    else:
        sigma = num_genes / num_tf
        err = sigma*jax.random.normal(key, shape=(num_samples,))
        y = y + err

    return X, y, betas, idx_on

def get_assoc_mat(*, num_tf, num_genes, corr=1.):
    feats = num_tf + (num_tf * num_genes)
    assoc_mat = np.eye(feats, feats)
    m = num_genes + 1
    for t in range(0, m * num_tf, m):
        for g in range(t + 1, t + m):
            assoc_mat[t, g] = corr
            assoc_mat[g, t] = corr
    return assoc_mat

def assign_cols(X, append_y=True):
    X_copy = X.copy()
    cols = []
    if append_y:
        for i in range(X_copy.shape[1] - 1):
            cols.append(f"f{i + 1}")

        cols.append("y")
    else:
        for i in range(X_copy.shape[1]):
            cols.append(f"f{i + 1}")
    X_copy.columns = cols

    return X_copy



def prepare_data(seeds, seed_idx, data, nets, out_val_size=0.2, test_size=0.2):
    seed = seeds[seed_idx]
    X, y = data[seed_idx].iloc[:,:-1].to_numpy().astype(np.float), data[seed_idx].iloc[:,-1].to_numpy().astype(np.float)
    # print(np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, shuffle=True, stratify=y,
                                                        test_size=test_size)
    X_train, X_out_val, y_train, y_out_val = train_test_split(X_train, y_train, random_state=seed,
                                                              shuffle=True, stratify=y_train,
                                                              test_size=out_val_size)
    net = nets[seed_idx].to_numpy()
    return seed, net, (X_train, X_out_val, X_test, y_train, y_out_val, y_test)

def load_bmm_files(parent_dir):
    net_dir = os.path.join(parent_dir, "net")
    feat_dir = os.path.join(parent_dir, "feats")
    data_dir = os.path.join(parent_dir, "data")

    net_dfs = []
    data_dfs = []
    feat_ls = []

    with open(os.path.join(parent_dir, "rand_seeds.txt"), "r") as fp:
        seed_str = fp.readline().strip()

    seeds = [int(s) for s in seed_str.split(',')]

    for s in seeds:
        data_df = pd.read_csv(os.path.join(data_dir, f"data_bm_{s}.csv"), header=None)
        net_df = pd.read_csv(os.path.join(net_dir, f"feat_net_{s}.csv"), header=None)
        with open(os.path.join(feat_dir, f"feats_{s}.txt"), "r") as fp:
            feats_str = fp.readline().strip()

        feats = [int(f) for f in feats_str.split(',')]

        data_df = assign_cols(data_df)

        data_dfs.append(data_df)
        net_dfs.append(net_df)
        feat_ls.append(feats)


    return seeds, data_dfs, net_dfs, feat_ls

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyData(data.Dataset):

    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.size = X.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x, y = self.data[idx], self.target[idx]
        return x, y

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)

def fpkm_to_expr(data_path, gene_id_data, idx_col="model_id"):
    rna_seq_data = pd.read_csv(data_path)
    #Calculate log2 transformed values of TPM (Transcripts per Million) form fpkm
    tpm_data = rna_seq_data.groupby([idx_col])["fpkm"].transform(lambda x : np.log2(((x  / x.sum()) * 1e6) + 1)) # add pseudp-count of 1 to avoid taking log2(0)
    rna_seq_data["log2.tpm"] = tpm_data
    rna_seq_data["idx"] = rna_seq_data.groupby(idx_col).cumcount()
    exp_data = rna_seq_data.pivot(index=idx_col ,columns="idx", values="log2.tpm")
    gene_sym_data = pd.read_csv(gene_id_data, index_col="gene_id")
    gene_ids = rna_seq_data["gene_id"].unique()
    gene_syms = gene_sym_data.loc[gene_ids]["hgnc_symbol"]
    exp_data.columns = gene_syms
    return exp_data


def preprocess_data(seed, X, y, transformer=None, val_size=0.2, test_size=0.2, reset_index=False):
    X_train_outer_df, X_test_df, y_train_outer_df, y_test_df = train_test_split(X, y, random_state=seed,
                                                                                shuffle=True, test_size=test_size)

    if transformer is not None:
        train_transformer = transformer.fit(X_train_outer_df)

        train_transformed = train_transformer.transform(X_train_outer_df)
        test_transformed = train_transformer.transform(X_test_df)
        X_train_outer_df = pd.DataFrame(train_transformed, columns=X_train_outer_df.columns)
        X_test_df = pd.DataFrame(test_transformed, columns=X_test_df.columns)

    if reset_index:
       X_train_outer_df, y_train_outer_df = X_train_outer_df.reset_index().drop("index", axis=1), \
                                            y_train_outer_df.reset_index().drop("index", axis=1).squeeze()

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_train_outer_df, y_train_outer_df, shuffle=True,
                                                                  random_state=seed, test_size=val_size)
    train_indices, val_indices = X_train_df.index.to_list(), X_val_df.index.to_list()

    X_train_outer, y_train_outer = X_train_outer_df.values, y_train_outer_df.values
    X_train, y_train = X_train_df.values, y_train_df.values
    X_val, y_val = X_val_df.values, y_val_df.values
    X_test, y_test = X_test_df.values, y_test_df.values

    return X_train_outer, X_train, X_val, X_test, y_train_outer, y_train, y_val, y_test, (train_indices, val_indices)
