#include "channel.h"

//Buffer for derivatives

double2* LDIAG;
double2* UDIAG;
double2* CDIAG;
double2* AUX;
float2* ddv;
float2* g;

int main(int argc, char** argv)
{ 	
  int rank;
  int size;
  int i;
  int iglobal;
  cudaDeviceProp prop;//设备属性内置结构体类型Struct
  int Ndevices;
  domain_t domain = {0, 0, 0, 0, 0, 0};
  char *ginput, *goutput, *ddvinput, *ddvoutput;
  config_t config;  //libconfig.h
  paths_t path;
  
  MPI_Init(&argc, &argv);	//初始化  可以不加参数
  H5open();  //初始化
  
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initial configuration // 源自config库
  if(rank == 0){
    config = read_config_file("run.conf");
    //printf("Reading configuration\n");
    read_domain_from_config(&domain,&config);
    //printf("Reading input file names\n");
    read_filenames_from_config(&path,&config);
  }
    //    int MPIAPI MPI_Bcast(
    //            _Inout_  void        *buffer,
    //            _In_    int          count,
    //            _In_    MPI_Datatype datatype,
    //    _In_    int          root,
    //            _In_    MPI_Comm     comm
    //    );
  MPI_Bcast(&(domain.nx), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.ny), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.nz), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.lx), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.lz), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.reynolds), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(domain.massflux), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.nsteps), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.freq_stats), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.ginput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.goutput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.ddvinput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.ddvoutput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.umeaninput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.umeanoutput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.umeaninput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.umeanoutput), 100, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(path.path), 100, MPI_CHAR, 0, MPI_COMM_WORLD);

  domain.rank = rank;
  domain.size = size;
  domain.iglobal = domain.nx * domain.rank / domain.size;//广义的子域起始索引    MPITUTORIAL随机游走案例
/* 
  for (i=0; i<domain.size; i++){
    if (i == domain.rank){
      if (i == 0) printf("rank, size, iglobal, nx, ny ,nz\n");
      if (i == 0) printf("===============================\n");
      MPI_Barrier(MPI_COMM_WORLD);
      printf("%d, %d, %d, %d, %d, %d\n",
	     domain.rank, domain.size, domain.iglobal,
	     domain.nx, domain.ny, domain.nz);
      fflush(stdout);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
*/  
  if(size != MPISIZE){
    printf("Error. The number of MPI processes is hardcoded in channel.h. Got%d, expecting %d\n",
	   size,
	   MPISIZE);
    exit(1);
  }
  
  CHECK_CUDART( cudaGetDeviceCount(&Ndevices));
  //printf("Ndevices=%d\n",Ndevices);
  
  //Local id
  iglobal=NXSIZE*rank;//局部的子域x方向索引
  //printf("(SIZE,RANK)=(%d,%d)\n",size,rank);
  
  CHECK_CUDART( cudaGetDeviceProperties(&prop,0) );
/*  
  if(rank == 0){
    // ACHTUNG! Please, nVidia, look at this hard limitation.
    if(prop.maxThreadsPerBlock < NY){
      printf("Too many points in the wall-normal direction\n");
      exit(-1);
    }
    //printf("MaxthreadperN=%d\n",prop.maxThreadsPerBlock);
  }
*/

  // Set up cuda device
  //cudaCheck(cudaSetDevice(rank),domain,"Set");		

  //Set the whole damn thing up
  setUp(domain);
  
  if(rank==0){
    setRKmean();
  }

  if(rank == 0){	
    printf("\nAllocation...\n");
  }

  //Allocate initial memory
  //Two buffers allocated
  //cudaCheck(cudaMalloc(&ddv,SIZE),domain,"malloc");
  //cudaCheck(cudaMalloc(&g,SIZE),domain,"malloc");
  
  //Read data
  if(strcmp(path.ginput,"-") == 0){
    if(rank == 0) printf("No input files specified. Creating empty files\n");
    genRandData(ddv,g,(float) NX,domain);
  }
  else{
    if(rank == 0) printf("Reading Data...\n");
    readData(ddv,g,path,domain);
  }
  //scale(ddv,10.0f);scale(g,10.0f);
  //genRandData(ddv,g,(float)(NX*NZ),domain);

  
  if(rank == 0){
    if(strcmp(path.umeaninput,"-")!= 0){
      readU(path.umeaninput);
      readW(path.wmeaninput);
    }
  }
  
  /*
    checkDerivatives();
    checkHemholzt();
    checkImplicit();*/
    //checkImplicit(domain);
    //checkBilaplacian(domain);
  

  if(rank == 0){	
    printf("Starting RK iterations...\n\n");
  }

  RKstep(ddv,g,1,domain,path);
  
  //Write data
  
  writeData(ddv,g,path,domain);
  
  if(rank==0){
    writeU(path.umeanoutput);
    writeW(path.wmeanoutput);
    //config_destroy(&config);
  }

  H5close();
  MPI_Finalize();
  
  return 0;
}



