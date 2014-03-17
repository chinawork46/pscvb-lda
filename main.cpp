#include "scvb.h"
#include <cstdlib>
#include <cstdio>

int main(int argc, char* argv[])
{
	SCVB* scvb;
	const char* doc_filename;
	char buffer[512];
	int k_topics = 100;
	int m_batchsize = 100;
	int iterations = 5;
	int words_to_output = 100;

	if (argc > 3)
	{
		doc_filename = argv[1];
		iterations = atoi(argv[2]);
		k_topics = atoi(argv[3]);
	}
	else
		doc_filename = "data/docword.nips.txt";

	scvb = new SCVB(doc_filename, k_topics, m_batchsize, words_to_output);

	for (int i = 0; i < iterations; ++i)
	{
		printf("Iteration: %d\n", i);	
		scvb->run();
		printf("Perplexity: %lf\n", scvb->perplexity());
	}
	
	sprintf(buffer, "nips_%d.txt", iterations);
	scvb->write_to_file(buffer);
	scvb->write_output_files();
	
	return 0;
}