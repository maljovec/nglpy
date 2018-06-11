#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

__global__
void prune_discrete(const int N, const int D, const int K, const int steps,
                    float *X, int *edgesIn, int *edgesOut, float *erTemplate)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    
    // References to points in X
    float *p, *q, *r;

    //TODO: Fix this
    // Computed vectors representing the edge under test pq and the vector from
    // one end point to a third point r (We will iterate over all possible r's)
    float pq[2] = {};
    float pr[2] = {};

    // Different iterator/indexing variables i, j, and n are rows in X
    // representing p, q, and r, respectively
    // k is the nearest neighbor, d is the dimension
    int i, j, k, d, n;

    // t is the parameterization of the projection of pr onto pq
    // In layman's terms, this is the length of the shadow pr casts onto pq
    // lookup is the 
    float t;
    int lookup;

    // Some other temporary variables
    float length_squared;
    float squared_distance_to_edge;
    float minimum_allowable_distance;

    for (k = index_y; k < K; k += stride_y) {
        for (i = index_x; i < N; i += stride_x) {

            p = &(X[D*i]);
            j = edgesIn[K*i+k];
            q = &(X[D*j]);
        
            length_squared = 0;
            for(d = 0; d < D; d++){
                pq[d] = p[d] - q[d];
                length_squared += pq[d]*pq[d];
            }
            // A point should not be connected to itself
            if(length_squared == 0) {
                edgesOut[K*i+k] = -1;
                continue;
            }
            
            for(n = 0; n < N; n++) {
                t = 0;
                r = &(X[D*n]);

                for(d = 0; d < D; d++){
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;
                lookup = __float2int_rd(abs(steps * (2 * t - 1))+0.5);
                if (lookup >= 0 && lookup <= steps) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++){
                        squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                    }    
                    minimum_allowable_distance = sqrt(length_squared)*erTemplate[lookup];

                    if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                        edgesOut[K*i+k] = -1;
                        break;
                    }
                }
            }
        }
    }
}

__global__
void prune(const int N, const int D, const int K, const int steps, float *X,
           int *edgesIn, int *edgesOut, float lp, float beta)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;

    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    
    float *p, *q, *r;

    float pq[2] = {};
    float pr[2] = {};

    int i, j, k, d, n;
    float t;

    float length_squared;
    float squared_distance_to_edge;
    float minimum_allowable_distance;

    ////////////////////////////////////////////////////////////
    float xC, yC, radius, y;
    ////////////////////////////////////////////////////////////

    for (k = index_y; k < K; k += stride_y) {
        for (i = index_x; i < N; i += stride_x) {
            p = &(X[D*i]);
            j = edgesIn[K*i+k];
            q = &(X[D*j]);
        
            length_squared = 0;
            for(d = 0; d < D; d++){
                pq[d] = p[d] - q[d];
                length_squared += pq[d]*pq[d];
            }
            // A point should not be connected to itself
            if(length_squared == 0) {
                edgesOut[K*i+k] = -1;
                continue;
            }
            
            for(n = 0; n < N; n++) {
            // for(n = 1; n < 1+1; n++) {
                r = &(X[D*n]);

                // t is the parameterization of the projection of pr onto pq
                // In layman's terms, this is the length of the shadow pr casts onto pq
                t = 0;
                for(d = 0; d < D; d++){
                    pr[d] = p[d] - r[d];
                    t += pr[d]*pq[d];
                }

                t /= length_squared;

                if (t > 0 && t < 1) {
                    squared_distance_to_edge = 0;
                    for(d = 0; d < D; d++){
                        squared_distance_to_edge += (pr[d] - pq[d]*t)*(pr[d] - pq[d]*t);
                    }

                    ////////////////////////////////////////////////////////////
                    // ported from python function, can possibly be improved 
                    // in terms of performance
                    xC = 0;
                    yC = 0;

                    if (beta <= 1) {
                        radius = 1. / beta;
                        yC = powf(powf(radius, lp) - 1, 1. / lp);
                    }
                    else {
                        radius = beta;
                        xC = 1. - beta;
                    }
                    t = fabs(2*t-1);
                    y = powf(powf(radius, lp) - powf(t-xC, lp), 1. / lp) - yC;
                    minimum_allowable_distance = 0.5*y*sqrt(length_squared);

                    //////////////////////////////////////////////////////////
                    if(sqrt(squared_distance_to_edge) < minimum_allowable_distance) {
                        edgesOut[K*i+k] = -1;
                        break;
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  std::cout << "using " << properties.multiProcessorCount << " multiprocessors"
            << std::endl 
            << "max threads per processor: " 
            << properties.maxThreadsPerMultiProcessor << std::endl;

  std::string pointFile = "../data_2_100_0.csv";
  std::string edgeFile = "../knn_2D_100.txt";
  std::string templateFile = "template.txt";
  int N = 100;
  int D = 2;
  int K = 100;
  int steps = 49;

  // Load data set and edges from files
  // TODO
  float *x;
  int *edgesIn;
  int *edgesOut;
  float *referenceShape;

  cudaMallocManaged(&x, N*D*sizeof(float));
  cudaMallocManaged(&edgesIn, N*K*sizeof(int));
  cudaMallocManaged(&edgesOut, N*K*sizeof(int));
  cudaMallocManaged(&referenceShape, (steps+1)*sizeof(float));

  std::ifstream file1( pointFile );
  
  int i = 0;
  int d = 0;
  std::string line;
  
  while ( std::getline(file1, line) )
  {
    std::istringstream iss(line);
    for (d = 0; d < D; d++) {
      iss >> x[i*D+d];
    }
    i++;
  }
  file1.close();
  
  std::ifstream file2 ( edgeFile );
  i = 0;
  int k = 0;
  while ( std::getline(file2, line) )
  {
    std::istringstream iss(line);
    for (k = 0; k < K; k++) {
      iss >> edgesIn[i*K+k];
      edgesOut[i*K+k] = edgesIn[i*K+k];
    }
    i++;
  }
  file2.close();

  std::ifstream file3 ( templateFile );
  i=0;
  while ( std::getline(file3, line) )
  {
    std::istringstream iss(line);
    iss >> referenceShape[i++];
  }
  file3.close();

  // Run kernel on the GPU
  dim3 blockSize(32, 32);
  dim3 gridSize(4, 4);
//   prune_discrete_test<<<gridSize, blockSize>>>(N, D, K, steps, x, edgesIn, edgesOut, referenceShape);
  prune<<<gridSize, blockSize>>>(N, D, K, steps, x, edgesIn, edgesOut, 2, 1);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));

  cudaDeviceSynchronize();
  
  std::ofstream file4 ( "output.txt" );
  for(i = 0; i < N; i++) {
    std::string sep = "";
    for(k = 0; k < K; k++) {
      file4 << sep << edgesOut[i*K+k];
      sep = " ";
    }
    file4 << std::endl;
  }
  
  std::ofstream file5 ( "/home/maljovec/projects/active/ngl/edges_2D_gpu.txt" );
  for(i = 0; i < N; i++) {
    for(k = 0; k < K; k++) {
      if (edgesOut[i*K+k] != -1) {
        file5 << i << " " << edgesOut[i*K+k] << std::endl;
      }
    }
  }

  // Free memory
  cudaFree(x);
  cudaFree(edgesIn);
  cudaFree(edgesOut);

  return 0;
}