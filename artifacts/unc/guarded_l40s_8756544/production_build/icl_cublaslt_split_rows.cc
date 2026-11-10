#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
extern "C" int try_split_rows(void* x,void* w,void* bias,void* out,void* work,size_t bytes,void* stream,int idx,int split,int scheme,int rows,int* count,int* picked) {
 cublasLtHandle_t h=nullptr; cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr; cublasStatus_t s;
 #define C(call) do{s=(call);if(s!=CUBLAS_STATUS_SUCCESS) return (int)s;}while(0)
 C(cublasLtCreate(&h)); C(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
 cublasOperation_t trans=CUBLAS_OP_T; C(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 cublasLtEpilogue_t epi=CUBLASLT_EPILOGUE_BIAS; C(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi))); C(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));
 C(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,512,512,512)); C(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,512,rows,512)); C(cublasLtMatrixLayoutCreate(&c,CUDA_R_16F,512,rows,512));
 int ids[256],n=0,valid=0; C(cublasLtMatmulAlgoGetIds(h,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,256,ids,&n)); cublasLtMatmulAlgo_t selected; bool found=false;
 for(int i=0;i<n;i++){cublasLtMatmulAlgo_t algo; cublasLtMatmulHeuristicResult_t res;
  if(cublasLtMatmulAlgoInit(h,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,CUDA_R_16F,ids[i],&algo)!=CUBLAS_STATUS_SUCCESS)continue;
  uint32_t tile=CUBLASLT_MATMUL_TILE_256x128,stage=CUBLASLT_MATMUL_STAGES_32x3;
  if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS)continue;
  if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stage,sizeof(stage))!=CUBLAS_STATUS_SUCCESS)continue;
  if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))!=CUBLAS_STATUS_SUCCESS)continue;
  uint32_t reduce=(uint32_t)scheme; if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduce,sizeof(reduce))!=CUBLAS_STATUS_SUCCESS)continue;
  if(cublasLtMatmulAlgoCheck(h,op,a,b,c,c,&algo,&res)!=CUBLAS_STATUS_SUCCESS || res.workspaceSize>bytes)continue;
  if(valid==idx){selected=algo;*picked=ids[i];found=true;} valid++;
 }
 *count=valid; if(!found){s=CUBLAS_STATUS_NOT_SUPPORTED;}else{float alpha=1,beta=0;s=cublasLtMatmul(h,op,&alpha,w,a,x,b,&beta,out,c,out,c,&selected,work,bytes,(cudaStream_t)stream);}
 cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);return (int)s;
}
