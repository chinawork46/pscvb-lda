#include <vector>
#include <map>

#define NUM_THREADS 16
#define LOCK_NUM 1024
#define BIT_MASK 1023
#define THREADING
#define CLUMPING
#ifdef THREADING
#include <omp.h>
#endif

using namespace std;

typedef float SVB_FLOAT;

class SCVB
{
	protected:
		struct word_id_count
		{
			int word_id;
			int word_count;
		};

		int k_topics;
		int m_batchsize;
		int corpus_count;
		int burn_in_passes;
		SVB_FLOAT phi_tau;
		SVB_FLOAT theta_tau;
		SVB_FLOAT rho_s;
		SVB_FLOAT rho_phi_t;

#ifdef THREADING
		int num_threads;

		//lock for synchronization 
		omp_lock_t word_lock[LOCK_NUM];

		//thread specific data
		SVB_FLOAT** thread_N_z_caps;
		SVB_FLOAT** local_gamma_ijs;
		int* rho_theta_ts;
#else
		SVB_FLOAT* gamma_ij;
		SVB_FLOAT rho_theta_t;
#endif

		SVB_FLOAT kappa;
		vector<word_id_count>** doc_wordid;
		int* Cj;
		int doc_count;
		int total_count;

		SVB_FLOAT alpha;
		SVB_FLOAT eta;
		SVB_FLOAT sum_eta;

		SVB_FLOAT* N_z;
		SVB_FLOAT** N_phi;
		SVB_FLOAT** N_theta;

		//temporary variables, allocated ones, uses many times
		SVB_FLOAT* N_z_cap;
		SVB_FLOAT** N_phi_cap;
		vector<int>* doc_ids;

		void init_docs(const char* file_name, vector<word_id_count>*** pdocs, int** pCj);

		void Equation5(int word_id, int doc_id, SVB_FLOAT* gamma_ij);
#ifdef CLUMPING
		void Equation6(int Cj, SVB_FLOAT* gamma_ij, SVB_FLOAT rho_theta, int doc_id, int word_count);
#else
		void Equation6(int Cj, SVB_FLOAT* gamma_ij, SVB_FLOAT rho_theta, int doc_id);
#endif
		void Equation7(SVB_FLOAT** N_phi_cap, SVB_FLOAT rho_phi, double mult);
		void Equation8(SVB_FLOAT* Nz_cap, SVB_FLOAT rho_phi);

		void vector_mul(SVB_FLOAT* v, SVB_FLOAT m, int n);
		void vector_add(SVB_FLOAT* v, SVB_FLOAT a, int n);
		void zero_2dim(SVB_FLOAT** d, int start, int row_count, int col_count);

		inline SVB_FLOAT get_rho_phi();

#ifdef THREADING
		inline SVB_FLOAT get_rho_theta(int* theta_t);
#else
		inline SVB_FLOAT get_rho_theta();
#endif	

		void reset_rho_phi_t();
		void reset_rho_theta_t();

	public:
		SCVB(const char* docs_file, int k_topics, int m_batchsize);
		~SCVB();

		void miniBatch(int*  doc_ids, int n_docs);
		
		void run();
		void write_to_file(char* file_name);
		void write_output_files();
		SVB_FLOAT perplexity();
};
