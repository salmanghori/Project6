/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.8 2007/02/19 23:57:39 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <limits.h>

// Windows OpenMP and timer stuff  MW 9/7/07
# include <Windows.h>
# include <tchar.h>
//# include <process.h>
# include <omp.h>
# include <mmsystem.h>
# include <time.h>

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

 /*# define N	10000000*/

# define N 30000000

# define NTIMES	40
# define OFFSET	0
# define WOWDIRLEN 1024

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

 //Check
static double	a[N + OFFSET],
b[N + OFFSET],
c[N + OFFSET];


//Check
static float	a_1[N + OFFSET],
b_1[N + OFFSET],
c_1[N + OFFSET];

static double	avgtime[4] = { 0 }, maxtime[4] = { 0 },
mintime[4] = { FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX };

static const char* label[4] = { "MUL_DOUBLE:      ", "MUL_FLOAT_:     ","Add:       ", "Triad:     " };

static double	bytes[4] = {
	2 * sizeof(double) * N,
	2 * sizeof(double) * N,
	3 * sizeof(double) * N,
	3 * sizeof(double) * N
};

#define Cvt_Win32LI_to_Dbl(x) ((x.HighPart * 4294967296.0) + x.LowPart)
static LARGE_INTEGER PerfCtrFreq;
static double PerfCtrFreq_asDbl = 0.0;

void PrintOSVersion(OSVERSIONINFO);
void OS64bitchk(int* pIs64BitOS);

extern double mysecond();
extern void checkSTREAMresults();
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(double scalar);
#endif

int
main()
{
	int			quantum, checktick();
	int			BytesPerWord;
	register int	j, k;
	double		scalar, t, times[4][NTIMES];
	OSVERSIONINFO OSVersion;
	BOOL brc;
	/*int Is64bitOS = 0;*/
	//================================================================
	/* Get Windows OS version */
	OSVersion.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
	GetVersionEx(&OSVersion);
	PrintOSVersion(OSVersion);
	/*
	OS64bitchk (&Is64bitOS);
	if (Is64bitOS != 0) {
	   printf("Detected 32 bit OS.\n");
	}
	else {
	   printf("Detected 64 bit OS.\n");
	}
	*/
	//================================================================
	//================================================================
#ifndef _OPENMP
	HANDLE hThisProcess;
	DWORD_PTR OldProcAffMask, OldSysAffMask, NewProcAffMask;

	/* bind to cpu 0 if not OpenMP */
	hThisProcess = GetCurrentProcess();
	brc = GetProcessAffinityMask(hThisProcess,
		&OldProcAffMask,
		&OldSysAffMask);
	printf("Old Process Affinity Mask = 0x%lx System Affinity Mask = 0x%lx\n", OldProcAffMask, OldSysAffMask);
	NewProcAffMask = 0x1;
	brc = SetProcessAffinityMask(hThisProcess,
		NewProcAffMask);
	printf("New Process Affinity Mask = 0x%lx\n", NewProcAffMask);
#endif
	//================================================================
	//================================================================
	/* --- SETUP --- determine precision and check timing --- */

	printf(HLINE);
	printf("STREAM version $Revision: 5.8 $\n");
	printf(HLINE);
	BytesPerWord = sizeof(double);
	printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
		BytesPerWord);

	printf(HLINE);
	printf("Array size = %d, Offset = %d\n", N, OFFSET);
	printf("Total memory required = %.1f MB.\n",
		(3.0 * BytesPerWord) * ((double)N / 1048576.0));
	printf("Each test is run %d times, but only\n", NTIMES);
	printf("the *best* time for each is used.\n");

#ifdef _OPENMP
	printf(HLINE);
#pragma omp parallel 
	{
#pragma omp master
		{
			k = omp_get_num_threads();
			printf("Number of Threads requested = %i\n", k);
		}
	}
#endif

	printf(HLINE);
#pragma omp parallel
	{
		printf("Printing one line per active thread....\n");
	}
	//================================================================

	//================================================================

	/* Get initial value for system clock. */
#pragma omp parallel for
	for (j = 0; j < N; j++) {
		a[j] = 1.0;
		b[j] = 2.0;
		c[j] = 0.0;
	}

	printf(HLINE);
	brc = QueryPerformanceFrequency(&PerfCtrFreq);
	if (brc == 0) {
		printf("Error returned from call to QueryPerformanceFrequency\n");
		exit(-1);
	}
	PerfCtrFreq_asDbl = Cvt_Win32LI_to_Dbl(PerfCtrFreq);

	if ((quantum = checktick()) >= 1)
		printf("Your clock granularity/precision appears to be "
			"%d microseconds.\n", quantum);
	else {
		printf("Your clock granularity appears to be "
			"less than one microsecond.\n");
		quantum = 1;
	}

	t = mysecond();
#pragma omp parallel for
	for (j = 0; j < N; j++)
		a[j] = 2.0E0 * a[j];
	t = 1.0E6 * (mysecond() - t);

	printf("Each test below will take on the order"
		" of %d microseconds.\n", (int)t);
	printf("   (= %d clock ticks)\n", (int)(t / quantum));
	printf("Increase the size of the arrays if this shows that\n");
	printf("you are not getting at least 20 clock ticks per test.\n");

	printf(HLINE);

	printf("WARNING -- The above is only a rough guideline.\n");
	printf("For best results, please be sure you know the\n");
	printf("precision of your system timer.\n");
	printf(HLINE);
	//================================================================



	 //Check

	//================================================================
	/*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
	//scalar = 3.0;
	for (k = 0; k < NTIMES; k++)
	{
		times[0][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Mul();
#else
#pragma omp parallel for
		for (j = 0; j < N; j++)
			c[j] = a[j] * b[j];
#endif
		times[0][k] = mysecond() - times[0][k];



		times[1][k] = mysecond();
#ifdef TUNED
		tuned_STREAM_Mul_1();
#else
#pragma omp parallel for
		for (j = 0; j < N; j++)
			c_1[j] = a_1[j] * b_1[j];
#endif
		times[1][k] = mysecond() - times[1][k];
	}

	/*	--- SUMMARY --- */

	 //Check

	for (k = 1; k < NTIMES; k++) /* note -- skip first iteration */
	{
		for (j = 0; j < 2; j++)
		{
			avgtime[j] = avgtime[j] + times[j][k];
			mintime[j] = MIN(mintime[j], times[j][k]);
			maxtime[j] = MAX(maxtime[j], times[j][k]);
		}
	}

	printf("Function          Rate (MB/s)   Avg time     Min time     Max time\n");
	for (j = 0; j < 2; j++) {
		avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

#ifdef _OPENMP
		if (mintime[j] < 0.0) {
			printf("Negative time detected.\n");
			printf("Did you set MP_BIND=yes, MP_BLIST, and OMP_NUM_THREADS accordingly?\n");
		}
#endif

		printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
			1.0E-06 * bytes[j] / mintime[j],
			avgtime[j],
			mintime[j],
			maxtime[j]);
	}
	printf(HLINE);
	printf(HLINE);
	//================================================================
	return 0;
}

# define	M	20

int checktick()
{
	int		i, minDelta, Delta;
	double	t1, t2, timesfound[M];

	/*  Collect a sequence of M unique time values from the system. */

	for (i = 0; i < M; i++) {
		t1 = mysecond();
		while (((t2 = mysecond()) - t1) < 1.0E-6)
			;
		timesfound[i] = t1 = t2;
	}

	/*
	 * Determine the minimum difference between these M values.
	 * This result will be our estimate (in microseconds) for the
	 * clock granularity.
	 */

	minDelta = 1000000;
	for (i = 1; i < M; i++) {
		Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
		minDelta = MIN(minDelta, MAX(Delta, 0));
	}

	return(minDelta);
}



// modified for Windows-  still not working right on GH!  -MW 9/7/07
//
/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

   //#include <sys/time.h>

double mysecond()
{
	//        struct timeval tp;
	//        struct timezone tzp;
	//        int i;
	//        i = gettimeofday(&tp,&tzp);
	//        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );

	//	return( (double) (timeGetTime() / 1000.0) );
	LARGE_INTEGER ts;
	double ts_asDbl;
	//	ts = clock();
	QueryPerformanceCounter(&ts);
	ts_asDbl = Cvt_Win32LI_to_Dbl(ts);
	return (ts_asDbl / PerfCtrFreq_asDbl); /* return secs */
}


void tuned_STREAM_Mul()
{
	//Check
	int j;
#pragma omp parallel for
	for (j = 0; j < N; j++)
		c[j] = a[j] * b[j];
}

void tuned_STREAM_Mul_1()
{
	//Check
	int j;
#pragma omp parallel for
	for (j = 0; j < N; j++)
		c_1[j] = a_1[j] * b_1[j];
}


void PrintOSVersion(OSVERSIONINFO OSVersion) {

	switch (OSVersion.dwMajorVersion) {
	case 4:
		switch (OSVersion.dwMinorVersion) {
		case 0:
			printf("Windows NT4 or 95 detected.\n");
			break;
		case 10:
			printf("Windows 98 detected.\n");
			break;
		case 90:
			printf("Windows Me detected.\n");
			break;
		default:
			printf("Unknown Minor OS Version (4.%d)\n", OSVersion.dwMinorVersion);
			break;
		}
		break;
	case 5:
		switch (OSVersion.dwMinorVersion) {
		case 0:
			printf("Windows 2000 detected.\n");
			break;
		case 1:
			printf("Windows XP detected.\n");
			break;
		case 2:
			printf("Windows Server2003 or Server2003 R2 detected.\n");
			break;
		default:
			printf("Unknown Minor OS Version (5.%d)\n", OSVersion.dwMinorVersion);
			break;
		}
		break;
	case 6:
		switch (OSVersion.dwMinorVersion) {
		case 0:
			printf("Windows Vista or Server2008 detected.\n");
			break;
		default:
			printf("Unknown Minor OS Version (6.%d)\n", OSVersion.dwMinorVersion);
			break;
		}
		break;
	default:
		printf("OS Major-Minor Version not recognized (%d.%d)\n",
			OSVersion.dwMajorVersion, OSVersion.dwMinorVersion);
		break;
	}
	printf("OS Build number %d\n", OSVersion.dwBuildNumber);
	printf("Service Pack Level: %s\n", OSVersion.szCSDVersion);

}

