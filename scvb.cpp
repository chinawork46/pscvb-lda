#include "scvb.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;

SVB_FLOAT SCVB::get_rho_phi()
{
	SVB_FLOAT res;

	res = this->rho_s / pow(this->phi_tau + this->rho_phi_t, this->kappa);
	this->rho_phi_t += 1.0;

	return res;
}

#ifdef THREADING
SVB_FLOAT SCVB::get_rho_theta(int* rho_theta_t)
{
	SVB_FLOAT res;

	res = this->rho_s / pow(this->theta_tau + *rho_theta_t, this->kappa);
	*rho_theta_t += 1;

	return res;
}
#else
SVB_FLOAT SCVB::get_rho_theta()
{
	SVB_FLOAT res;

	res = this->rho_s / pow(this->theta_tau + this->rho_theta_t, this->kappa);
	this->rho_theta_t += 1.0;

	return res;
}
#endif
	
void SCVB::reset_rho_phi_t()
{
	this->rho_phi_t = 0.0;
}

void SCVB::reset_rho_theta_t()
{
#ifdef THREADING
	for (int i = 0; i < this->num_threads; ++i)
		this->rho_theta_ts[i] = 0.0;
#else
	this->rho_theta_t = 0.0;
#endif
}

void SCVB::vector_mul(SVB_FLOAT* v, SVB_FLOAT m, int n)
{
	int i;

	for (i = 0; i < n; ++i)
		v[i] *= m;
}

void SCVB::vector_add(SVB_FLOAT* v, SVB_FLOAT a, int n)
{
	int i;

	for (i = 0; i < n; ++i)
		v[i] += a;
}

void SCVB::Equation5(int word_id, int doc_id, SVB_FLOAT* gamma_ij)
{
	int k;
	SVB_FLOAT total = 0.0;

	for (k = 0; k < this->k_topics; ++k)
	{
		gamma_ij[k] = (this->N_phi[word_id][k] + this->eta) / (this->N_z[k] + this->sum_eta) * (this->N_theta[doc_id][k] + this->alpha);
		total += gamma_ij[k];
	}

	for (k = 0; k < this->k_topics; ++k)
		gamma_ij[k] /= total;
}

#ifdef CLUMPING
	void SCVB::Equation6(int Cj, SVB_FLOAT* gamma_ij, SVB_FLOAT rho_theta, int doc_id, int word_count)
	{
	    SVB_FLOAT a, m;
	    int i;
	    double m_rho_theta = pow((1 - rho_theta), word_count);

		for (i = 0; i < this->k_topics; ++i)
	    {
	    	a = (1 - m_rho_theta) * Cj * gamma_ij[i];
	    	m = m_rho_theta;
	    	this->N_theta[doc_id][i] = m * this->N_theta[doc_id][i] + a;
	    }
	}
#else
void SCVB::Equation6(int Cj, SVB_FLOAT* gamma_ij, SVB_FLOAT rho_theta, int doc_id)
{
    SVB_FLOAT a, m;
    int i;

	for (i = 0; i < this->k_topics; ++i)
    {
    	a = rho_theta * Cj * gamma_ij[i];
    	m = 1 - rho_theta;
    	this->N_theta[doc_id][i] = m * this->N_theta[doc_id][i] + a;
    }
}
#endif
		

void SCVB::Equation7(SVB_FLOAT** N_phi_cap, SVB_FLOAT rho_phi, double mult)
{
    SVB_FLOAT phi_mul = 1 - rho_phi;

#ifdef THREADING
    int total = this->corpus_count * this->k_topics;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < this->num_threads; ++i)
    {
    	int start = i * (total / this->num_threads);
    	int end = (i == this->num_threads - 1) ? total : (total / this->num_threads) * (i + 1);

    	for (int j = start; j < end; ++j)
    		this->N_phi[1][j] = phi_mul * this->N_phi[1][j] \
    							+ rho_phi * N_phi_cap[1][j] * mult;
    }

#else

    int word_iter, topic_iter;
	for (word_iter = 1; word_iter < this->corpus_count + 1; ++word_iter)
    	for (topic_iter = 0; topic_iter < this->k_topics; ++topic_iter)
    		this->N_phi[word_iter][topic_iter] = phi_mul * this->N_phi[word_iter][topic_iter] \
    											 + rho_phi * N_phi_cap[word_iter][topic_iter] * mult;
#endif
	
}
		

void SCVB::Equation8(SVB_FLOAT* Nz_cap, SVB_FLOAT rho_phi)
{
	SVB_FLOAT phi_mul = 1 - rho_phi;
	int topic_iter;

	for (topic_iter = 0; topic_iter < this->k_topics; ++topic_iter)
		this->N_z[topic_iter] = phi_mul * this->N_z[topic_iter] + rho_phi * Nz_cap[topic_iter];
}

void SCVB::init_docs(const char* file_name, vector<word_id_count>*** pdocs, int** pCj)
{
	FILE* fp = fopen(file_name, "r");
	int non_used;

	if (fp == NULL)
	{
		printf("Invalid filename.\nAborted\n");
		exit(-1);
	}

	//doc_count
	//corpus
	//total_words
	fscanf(fp, "%d", &this->doc_count);
	fscanf(fp, "%d", &this->corpus_count);
	fscanf(fp, "%d", &non_used);
	this->total_count = 0;

	//initialize docs, cj
	vector<word_id_count>** docs; 
	int* Cj;

	*pdocs = (vector<word_id_count>**)malloc(sizeof(vector<word_id_count>*) * (this->doc_count + 1));
	docs = *pdocs;
	memset(docs, 0, sizeof(vector<word_id_count>*) * (this->doc_count + 1));

	*pCj = (int*)malloc(sizeof(int) * (this->doc_count + 1));
	Cj = *pCj;
	memset(Cj, 0, sizeof(int) * (this->doc_count + 1));

	int doc_id, word_id, word_count;

	while (fscanf(fp, "%d%d%d", &doc_id, &word_id, &word_count) != EOF)
	{
		word_id_count wc;

		this->total_count += word_count;
		Cj[doc_id] += word_count;
		if (docs[doc_id] == NULL)
			docs[doc_id] = new vector<word_id_count>();

		wc.word_id = word_id;
		wc.word_count = word_count;
		docs[doc_id]->push_back(wc);
	}

	fclose(fp);
}

SCVB::~SCVB()
{
	free(this->N_z_cap);
}

SCVB::SCVB(const char* docs_file, int k_topics, int m_batchsize, int words_to_output)
{
	int i, doc_id, count;

	this->k_topics = k_topics;
	this->m_batchsize = m_batchsize;
	this->kappa = 0.9;
	this->burn_in_passes = 0;
	this->rho_s = 10;
	this->phi_tau = 100;
	this->theta_tau = 1000;
	this->rho_phi_t = 0.0;
	this->words_to_output = words_to_output;
	
	//initialize doc file and corpus_count
	init_docs(docs_file, &this->doc_wordid, &this->Cj);

	//initialize all the variables
	this->alpha = 0.1;
	this->eta = 0.01;
	this->sum_eta = 0.01 * this->corpus_count;
	this->doc_ids = new vector<int>(this->doc_count);
	for (i = 1; i < this->doc_count + 1; ++i)
		(*this->doc_ids)[i - 1] = i;

	this->N_z = (SVB_FLOAT*)calloc(this->k_topics, sizeof(SVB_FLOAT));
	
	this->N_theta = (SVB_FLOAT**)malloc(sizeof(SVB_FLOAT*) * (this->doc_count + 1));
	for (i = 1; i < this->doc_count + 1; ++i)
		this->N_theta[i] = (SVB_FLOAT*)calloc(this->k_topics, sizeof(SVB_FLOAT));

	this->N_phi = (SVB_FLOAT**)malloc(sizeof(SVB_FLOAT*) * (this->corpus_count + 1));
	SVB_FLOAT* N_phi_1dim = (SVB_FLOAT*)calloc(this->k_topics * this->corpus_count, sizeof(SVB_FLOAT));
	count = 0;

	for (i = 1; i < this->corpus_count + 1; ++i)
	{
		this->N_phi[i] = &N_phi_1dim[count];
		count += this->k_topics;
	}

	//initialize temporary variables
	this->N_phi_cap = (SVB_FLOAT**)malloc(sizeof(SVB_FLOAT*) * (this->corpus_count + 1));
	SVB_FLOAT* N_phi_cap_1dim = (SVB_FLOAT*)calloc(this->k_topics * this->corpus_count, sizeof(SVB_FLOAT));
	count = 0;

	for (i = 1; i < this->corpus_count + 1; ++i)
	{
		this->N_phi_cap[i] = &N_phi_cap_1dim[count];
		count += this->k_topics;
	}
	this->N_z_cap = (SVB_FLOAT*)calloc(this->k_topics, sizeof(SVB_FLOAT));

#ifdef THREADING
	//init thread specific data
	this->num_threads= NUM_THREADS;
	for (i = 0; i < LOCK_NUM; ++i)
		omp_init_lock(&this->word_lock[i]);

	this->thread_N_z_caps = (SVB_FLOAT**)malloc(sizeof(SVB_FLOAT*) * this->num_threads);
	this->local_gamma_ijs = (SVB_FLOAT**)malloc(sizeof(SVB_FLOAT*) * this->num_threads);

	for (i = 0; i < this->num_threads; ++i)
	{
		this->thread_N_z_caps[i] = (SVB_FLOAT*)malloc(sizeof(SVB_FLOAT) * this->k_topics);
		this->local_gamma_ijs[i] = (SVB_FLOAT*)malloc(sizeof(SVB_FLOAT) * this->k_topics);	
	}

	this->rho_theta_ts = (int*)malloc(sizeof(int) * this->m_batchsize);
	this->rho_theta_ts[0] = 0;
#else
	this->rho_theta_t = 0.0;
	this->gamma_ij = (SVB_FLOAT*)malloc(sizeof(SVB_FLOAT) * this->k_topics);
#endif

	//initialize all the variables with random values
	srand(time(NULL));
	for (doc_id = 1; doc_id < this->doc_count + 1; ++doc_id)
	{
		if (this->doc_wordid[doc_id] == NULL)
			continue;

		for (auto word_iter = this->doc_wordid[doc_id]->begin(); word_iter != this->doc_wordid[doc_id]->end(); ++word_iter)
		{
			word_id_count wc = *word_iter;
			int r;

			for (int count = 0; count < wc.word_count; ++count)
			{
				r = rand() % this->k_topics;
			
				this->N_phi[wc.word_id][r] += 1.0;
				this->N_theta[doc_id][r] += 1.0;
				this->N_z[r] += 1.0;
			}
		} //end iterations of words in a doc
	} //end iterations of docs
}

void SCVB::zero_2dim(SVB_FLOAT** d, int start, int row_count, int col_count)
{
	memset(d[1], 0, this->corpus_count * this->k_topics * sizeof(SVB_FLOAT));
	//	int i;
	//for (i = start; i < row_count; ++i)
	//	memset(d[i], 0, col_count * sizeof(SVB_FLOAT));
}


void SCVB::miniBatch(int* doc_ids, int n_docs)
{
	int doc_iter;
	int total_Cj = 0;

	//initialzie total word count for all docs
	for (int i = 0; i < n_docs; ++i)
		total_Cj += this->Cj[doc_ids[i]];
	//initialize N_phi_cap N_z_cap
	//zero_2dim(this->N_phi_cap, 1, this->corpus_count + 1, this->k_topics);
	memset(this->N_phi_cap[1], 0, this->corpus_count * this->k_topics * sizeof(SVB_FLOAT));
	memset(this->N_z_cap, 0, sizeof(SVB_FLOAT) * this->k_topics);

#ifdef THREADING
	for (int i = 0; i < this->num_threads; ++i)
		memset(this->thread_N_z_caps[i], 0, sizeof(SVB_FLOAT) * this->k_topics);

	//Should use n_docs rather than this->m_batchsize, since the last batch could be smaller
	//Calculate rho_theta_t for each thread, this is really imporatant
	for (int i = 1; i < n_docs; ++i)
	//@# The step-sizes in clumping and non-clumping should be figure out clearly later
#ifdef CLUMPING
		if (this->doc_wordid[doc_ids[i - 1]] == NULL)
			this->rho_theta_ts[i] = this->rho_theta_ts[i - 1];
		else	
			this->rho_theta_ts[i] = this->doc_wordid[doc_ids[i - 1]]->size() * (1 + this->burn_in_passes) 
									+ this->rho_theta_ts[i - 1];	
		
#else
		this->rho_theta_ts[i] = this->Cj[doc_ids[i - 1]] * (1 + this->burn_in_passes) 
								+ this->rho_theta_ts[i - 1];
#endif

	#pragma omp parallel for num_threads(NUM_THREADS)
#endif
	for (doc_iter = 0; doc_iter < n_docs; ++doc_iter)
	{
		int burn_in_iter;
		int doc_id = doc_ids[doc_iter];
		SVB_FLOAT* local_gamma_ij;

#ifdef THREADING
		int rho_theta_t = this->rho_theta_ts[doc_iter];
		int tid = omp_get_thread_num();

		local_gamma_ij = this->local_gamma_ijs[tid];
#else
		local_gamma_ij = this->gamma_ij;
#endif

		if (this->doc_wordid[doc_id] == NULL)
			continue;

		for (burn_in_iter = 0; burn_in_iter < this->burn_in_passes; ++burn_in_iter)
		{
			for (auto word_iter = this->doc_wordid[doc_id]->begin(); word_iter != this->doc_wordid[doc_id]->end(); ++word_iter)
			{
				word_id_count wc = *word_iter;
#ifndef CLUMPING
				for (int i = 0; i < wc.word_count; ++i)
				{
#endif
#ifdef THREADING
				SVB_FLOAT rho_theta = get_rho_theta(&rho_theta_t);
#else
				SVB_FLOAT rho_theta = get_rho_theta();
#endif
					Equation5(wc.word_id, doc_id, local_gamma_ij);
#ifdef CLUMPING
					Equation6(this->Cj[doc_id], local_gamma_ij, rho_theta, doc_id, wc.word_count);
#else
					Equation6(this->Cj[doc_id], local_gamma_ij, rho_theta, doc_id);
#endif		
		
#ifndef CLUMPING
				} //end for word_count
#endif
			}
		}

		for (auto word_iter = this->doc_wordid[doc_id]->begin(); word_iter != this->doc_wordid[doc_id]->end(); ++word_iter)
		{
			word_id_count wc = *word_iter;

#ifndef CLUMPING
			for (int i = 0; i < wc.word_count; ++i)
			{
#endif

#ifdef THREADING
				SVB_FLOAT rho_theta = get_rho_theta(&rho_theta_t);
#else
				SVB_FLOAT rho_theta = get_rho_theta();
#endif
				Equation5(wc.word_id, doc_id, local_gamma_ij);

#ifdef CLUMPING
				Equation6(this->Cj[doc_id], local_gamma_ij, rho_theta, doc_id, wc.word_count);
#else
				Equation6(this->Cj[doc_id], local_gamma_ij, rho_theta, doc_id);
#endif
#ifdef THREADING
				int lock_id = wc.word_id & BIT_MASK;
				omp_set_lock(&this->word_lock[lock_id]);
				for (int k = 0; k < this->k_topics; ++k)
				{
#ifdef CLUMPING
					this->N_phi_cap[wc.word_id][k] += local_gamma_ij[k] * wc.word_count;
#else
					this->N_phi_cap[wc.word_id][k] += local_gamma_ij[k];
#endif			
				}				
				omp_unset_lock(&this->word_lock[lock_id]);
				
				for (int k = 0; k < this->k_topics; ++k)
				{
#ifdef CLUMPING
					this->thread_N_z_caps[tid][k] += local_gamma_ij[k] * wc.word_count;
#else
					this->thread_N_z_caps[tid][k] += local_gamma_ij[k];
#endif
				}	
#else
				for (int k = 0; k < this->k_topics; ++k)
				{
#ifdef CLUMPING
					this->N_phi_cap[wc.word_id][k] += local_gamma_ij[k]* wc.word_count;
					this->N_z_cap[k] += local_gamma_ij[k] * wc.word_count;
#else
					this->N_phi_cap[wc.word_id][k] += local_gamma_ij[k];
					this->N_z_cap[k] += local_gamma_ij[k];
#endif				
					
				}
#endif

#ifndef CLUMPING
			} //end for word_count
#endif
		}

	}

#ifdef THREADING
	for (int i = 0; i < this->num_threads; ++i)
		for (int k = 0; k < this->k_topics; ++k)
			this->N_z_cap[k] += this->thread_N_z_caps[i][k];
	//@# ?
	this->rho_theta_ts[0] = total_Cj;
#endif
	//Multiplier the normalizer for the caps
	SVB_FLOAT mult = (SVB_FLOAT)this->total_count / (SVB_FLOAT)total_Cj;
	//for (int i = 1; i < this->corpus_count + 1; ++i)
	//	vector_mul(this->N_phi_cap[i], mult, this->k_topics);

	vector_mul(this->N_z_cap, mult, this->k_topics);

	//Execute the update to N_phi and N_z
	SVB_FLOAT rho_phi = get_rho_phi();
	Equation7(this->N_phi_cap, rho_phi, mult);
	Equation8(this->N_z_cap, rho_phi);
}
		
void SCVB::run()
{
	int* docs = (int*)malloc(sizeof(int) * this->m_batchsize);
	int i;

	//init step
	reset_rho_theta_t();
	reset_rho_phi_t();

	srand(time(NULL));
	random_shuffle(this->doc_ids->begin(), this->doc_ids->end());

	for (i = 0; i < this->doc_ids->size();)
	{
		int count = this->m_batchsize;
		int t = 0;

		if (this->doc_ids->size() - i < this->m_batchsize)
			count = this->doc_ids->size() - i;

		for (int d = i; d < i + count; ++d)
		{
			docs[t] = (*this->doc_ids)[d];
			t += 1;
		}

		miniBatch(docs, count);
		i += count;
	}

	free(docs);
}

void SCVB::write_to_file(char* file_name)
{
	FILE* fp = fopen(file_name, "w");

	for (int i = 1; i < this->corpus_count + 1; ++i)
	{
		for (int k = 0; k < this->k_topics; ++k)
			fprintf(fp, "%lf ", this->N_phi[i][k]);
		fprintf(fp, "%s", "\n");
	}

	fclose(fp);
}

void SCVB::exchange(int* x1, int* x2)
{
	int temp = *x1;

	*x1 = *x2;
	*x2 = temp;
}

void SCVB::sort_col(int* idx_array, SVB_FLOAT** two_dim_data, int col_idx, int num_idx, bool desc)
{
	int comparator;
	int lidx = 1, ridx = num_idx - 1;

	if (num_idx < 2)
		return;

	comparator = idx_array[0];

	while (lidx < ridx)
	{
		while ((lidx < num_idx) &&
			   (two_dim_data[idx_array[lidx]][col_idx] > two_dim_data[comparator][col_idx]))
			++lidx;

		if ((lidx == num_idx) || (lidx >= ridx))
			break;

		while ((ridx > lidx) &&
			   (two_dim_data[idx_array[ridx]][col_idx] < two_dim_data[comparator][col_idx]))
			--ridx;

		if ((ridx == -1) || (ridx == lidx))
			break;

		exchange(&idx_array[lidx], &idx_array[ridx]);
		++lidx;
		--ridx;
	}

	if (lidx >= num_idx)
		lidx = num_idx - 1;

	while (two_dim_data[idx_array[lidx]][col_idx] < two_dim_data[comparator][col_idx])
		--lidx;

	exchange(&idx_array[0], &idx_array[lidx]);

	this->sort_col(&idx_array[0], two_dim_data, col_idx, lidx, desc);
	this->sort_col(&idx_array[lidx + 1], two_dim_data, col_idx, num_idx - lidx - 1, desc);
}	

void SCVB::write_output_files()
{
	const char* topic_file = "topics.txt";
	const char* doctopic_file = "doctopic.txt";
	int* idxArray = (int*)malloc(sizeof(int) * this->corpus_count);

	FILE* fp = fopen(topic_file, "w");

	for (int k = 0; k < this->k_topics; ++k)
	{
		for (int i = 0; i < this->corpus_count; ++i)
			idxArray[i] = i + 1;

		this->sort_col(idxArray, this->N_phi, k, this->corpus_count, true);

		SVB_FLOAT val;
		for (int i = 0; i < this->words_to_output - 1; ++i)
		{
			val = (this->N_phi[idxArray[i]][k] + this->eta) / (this->N_z[k] + this->corpus_count * this->eta);
			fprintf(fp, "%d:%lf, ", idxArray[i], val);
		}

		val = (this->N_phi[idxArray[this->words_to_output - 1]][k] + this->eta) / (this->N_z[k] + this->corpus_count * this->eta);
		fprintf(fp, "%d:%lf", idxArray[this->words_to_output - 1], val);

		fprintf(fp, "\n");
	}
	fclose(fp);

	free(idxArray);

	fp = fopen(doctopic_file, "w");

	for (int doc_iter = 1; doc_iter < this->doc_count + 1; ++doc_iter)
	{
		SVB_FLOAT norm = 0.0;

		for (int k = 0; k < this->k_topics; ++k)
			norm += this->N_theta[doc_iter][k];

		for (int k = 0; k < this->k_topics - 1; ++k)
			fprintf(fp, "%lf, ", (this->N_theta[doc_iter][k] + this->alpha) / (norm + this->alpha * this->k_topics));
		fprintf(fp, "%lf", (this->N_theta[doc_iter][this->k_topics - 1] + this->alpha) / (norm + this->alpha * this->k_topics));

		fprintf(fp, "\n");
	}
	fclose(fp);
}

SVB_FLOAT SCVB::perplexity()
{
	SVB_FLOAT res = 0.0;

	for (int d = 1; d < this->doc_count + 1; ++d)
	{
		SVB_FLOAT theta_d = 0.0;

		if (this->doc_wordid[d] == NULL)
			continue;

		for (int k = 0; k < this->k_topics; ++k)
			theta_d += this->N_theta[d][k];

		for (auto word_iter = this->doc_wordid[d]->begin(); word_iter != this->doc_wordid[d]->end(); ++word_iter)
		{
			SVB_FLOAT sum_k = 0.0;
			word_id_count wc = *word_iter;
			for (int k = 0; k < this->k_topics; ++k)
			{
				SVB_FLOAT theta = (this->alpha + this->N_theta[d][k]) / (this->k_topics * this->alpha + theta_d);
				SVB_FLOAT phi = (this->eta + this->N_phi[wc.word_id][k]) / (this->corpus_count * this->eta + this->N_z[k]);

				sum_k += theta * phi;
			}

			res += log(sum_k) * wc.word_count;
		}
	}

	res = exp(-1.0 * res / this->total_count);

	return res;
}
