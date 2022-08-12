import numpy as np
from sklearn import mixture
import multiprocessing as mp

"""
This script is used for low pass allele likelihood simulation
input: high pass genotype, shape (n_sample, n_variant, 1(dosage))
output: low pass reference/alternative allele, shape (n_sample, n_variant, 2 (ref,alt))
"""

class simulator:
    def __init__(self):
        """
        reload pre-trained 1x gmm model for each genotype
        """
        # model directory
        self.model_dir = "/home/kchen/low_pass/autoencoder_imputation/gmm_model"

        # homozygous reference model
        self.means_homoref = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_means.npy")
        self.covar_homoref = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_covariances.npy")
        self.gmm_homoref = mixture.GaussianMixture(n_components = len(self.means_homoref), covariance_type='full')
        self.gmm_homoref.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covar_homoref))
        self.gmm_homoref.weights_ = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_weights.npy")
        self.gmm_homoref.means_ = self.means_homoref
        self.gmm_homoref.covariances_ = self.covar_homoref

        # heterozygous model
        self.means_het = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_means.npy")
        self.covar_het = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_covariances.npy")
        self.gmm_het = mixture.GaussianMixture(n_components = len(self.means_het), covariance_type='full')
        self.gmm_het.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covar_het))
        self.gmm_het.weights_ = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_weights.npy")
        self.gmm_het.means_ = self.means_het
        self.gmm_het.covariances_ = self.covar_het
        
        # homozygous alternative model
        self.means_homoalt = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_means.npy")
        self.covar_homoalt = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_covariances.npy")
        self.gmm_homoalt = mixture.GaussianMixture(n_components = len(self.means_homoalt), covariance_type='full')
        self.gmm_homoalt.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covar_homoalt))
        self.gmm_homoalt.weights_ = np.load(self.model_dir + "/" + "gmm_1x_homozygous_reference_weights.npy")
        self.gmm_homoalt.means_ = self.means_homoalt
        self.gmm_homoalt.covariances_ = self.covar_homoalt
    
    def chunks(self, n_data, n_threads):
        for i in range(0, len(n_data), n_threads):
            yield n_data[i:i + n_threads]
    
    def multithread_sampling(self, gt_arr):
        gt_flat = gt_arr.flatten()
        nrow, ncol = gt_arr.shape
        lowpass_gl = np.empty([len(gt_flat),2])
        homoref_idx = np.where(gt_flat == 0)
        het_idx = np.where(gt_flat == 1)
        homoalt_idx = np.where(gt_flat == 2)
        sample_homoref = self.gmm_homoref.sample(len(homoref_idx))
        sample_het = self.gmm_homoref.sample(len(het_idx))
        sample_homoalt = self.gmm_homoref.sample(len(homoalt_idx))
        lowpass_gl[homoref_idx] = sample_homoref[0]
        lowpass_gl[het_idx] = sample_het[0]
        lowpass_gl[homoalt_idx] = sample_homoalt[0]
        lowpass_gl = lowpass_gl.reshape((nrow,ncol,2))
        return lowpass_gl

    def simulate_allele(self, highpass_arr): 
        nproc = mp.cpu_count()
        pool = mp.pool.ThreadPool(nproc)
        results = pool.map(self.multithread_sampling, self.chunks(highpass_arr, nproc))
        pool.close()
        pool.join()
        results = [val for sublist in results for val in sublist]
        lowpass_gl = np.array(results)
        return lowpass_gl
        
        
        # loop through all variant (key), each variant has an genotype array
        # apply multithread_sampling on genotype array
        # 
#         n_homoref = highpass_gt_counter[0]
#         n_het = highpass_gt_counter[1]
#         n_homoalt = highpass_gt_counter[2]        
#         homoref_sampled = self.gmm_homoref.sample(n_homoref)
#         het_sampled = self.gmm_het.sample(n_het)
#         homoalt_sampled = self.gmm_homoalt.sample(n_homoalt)
        
#         return homoref_sampled, het_sampled, homoalt_sampled
    

if __name__ == "__main__":
    
    data = np.arange(8).reshape(4,2,1)
    print(data)
    
    
    
    