#include "ksgc_winograd.h"
#include <math.h>
#include <fstream>
#include <hls_math.h>
#include <ap_fixed.h>
#include <string.h>

using namespace std;


void LOAD_W1x1_S16(FIX_WT WBUF[16][64], FIX_WT weight_buf1[16][4],
					FIX_WT weight_buf2[16][4], int CI)
{
#pragma HLS array_partition variable=WBUF dim=1 complete

#pragma HLS array_partition variable=weight_buf1 dim=0 complete
#pragma HLS array_partition variable=weight_buf2 dim=0 complete

	for(int ci = 0; ci < 4; ci++) {
#pragma HLS unroll
		for(int co = 0; co < 16; co++) {
#pragma HLS unroll
			weight_buf1[co][ci] = WBUF[co][ci + CI];
			weight_buf2[co][ci] = WBUF[co][ci + CI+4];
		}
	}
}

void LOAD_W3x3_S16_my(FIX_WT WBUF3x3[16][64], FIX_WT W3x3[16][4][4],int cof)
{
#pragma HLS ARRAY_PARTITION variable=W3x3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=WBUF3x3 complete dim=1
int n=cof*16;
	for(int h=0;h<4;h++)
	{
		for(int w=0;w<4;w++)
		{
#pragma HLS PIPELINE II=1
			for(int c=0;c<16;c++)
			{
#pragma HLS UNROLL
				W3x3[c][h][w]=WBUF3x3[c][n+h*4+w];
			}
		}
	}
}

void load_bias_my(FIX_WT bias[16], FIX_WT bias_buf[16])
{
	for(int co = 0; co < 16; co++) {
#pragma HLS unroll
//#pragma HLS pipeline
		bias_buf[co] = bias[co];
	}
}


void C_BUF4_test(FIX_FM a1, FIX_FM a2, FIX_FM a3, FIX_FM a4, FIX_16_7 top[4])
{
#pragma HLS ARRAY_PARTITION variable=top complete dim=1
	top[0]=a1;
	top[1]=a2;
	top[2]=a3;
	top[3]=a4;
}


void BTd_Multip4_Gg_AT(FIX_FM_acc Y[2],  
			    FIX_WT Gg0,  FIX_16_7 d0,
			    FIX_WT Gg1,  FIX_16_7 d1,
				FIX_WT Gg2,  FIX_16_7 d2,
				FIX_WT Gg3,  FIX_16_7 d3)
{
	FIX_FM_acc UV0;
	FIX_FM_acc UV1;
	FIX_FM_acc UV2;
	FIX_FM_acc UV3;

	UV0 = (d0 - d2) * Gg0;
	UV1 = (d1 + d2) * Gg1;
	UV2 = (d2 - d1) * Gg2;
	UV3 = (d1 - d3) * Gg3;

	Y[0] = UV0 + UV1 + UV2;
	Y[1] = UV1 - UV2 - UV3; 
}



void A_OFM(FIX_FM_acc ATofm[4][2],
			FIX_FM ATofmA[2][2],
			FIX_WT bias)
{
//#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=ATofm complete dim=1
#pragma HLS ARRAY_PARTITION variable=ATofmA complete dim=1
	ATofmA[0][0]=ATofm[0][0]+ATofm[1][0]+ATofm[2][0]+bias;
	ATofmA[0][1]=ATofm[1][0]-ATofm[2][0]-ATofm[3][0]+bias;
	
	ATofmA[1][0]=ATofm[0][1]+ATofm[1][1]+ATofm[2][1]+bias;
	ATofmA[1][1]=ATofm[1][1]-ATofm[2][1]-ATofm[3][1]+bias;
}
void ifm_trans_1D(FIX_FM bottom[4],FIX_16_7 top[4])
{
#pragma HLS ARRAY_PARTITION variable=bottom complete dim=1
#pragma HLS ARRAY_PARTITION variable=top complete dim=1

	top[0]=bottom[0]-bottom[2];
	top[1]=bottom[1]+bottom[2];
	top[2]=bottom[2]-bottom[1];
	top[3]=bottom[1]-bottom[3];
}

void ifm_trans_2row(FIX_FM ifm1[4], FIX_FM ifm2[4], FIX_16_7 DATA1[4], FIX_16_7 DATA2[4])
{
#pragma HLS ARRAY_PARTITION variable=ifm1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=ifm2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=DATA1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=DATA2 complete dim=1

	ifm_trans_1D(ifm1, DATA1);
	ifm_trans_1D(ifm2, DATA2);
}


void share_1D2DWin_conv32(FIX_FM bottom1[32][16][16],
                          FIX_FM bottom2[32][16][16],
			  FIX_FM_acc top[64][16][16],
			  FIX_WT weights[16][64],
			  FIX_WT bias[4][16],
			  uint2 chout,
			  uint1 mode)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=bottom1 dim=1 complete
#pragma HLS array_partition variable=bottom2 dim=1 complete

#pragma HLS array_partition variable=weights dim=1 complete

FIX_WT weight_buf1[16][4];
#pragma HLS array_partition variable=weight_buf1 dim=1 complete
#pragma HLS array_partition variable=weight_buf1 dim=2 complete
FIX_WT weight_buf2[16][4];
#pragma HLS array_partition variable=weight_buf2 dim=1 complete
#pragma HLS array_partition variable=weight_buf2 dim=2 complete


FIX_16_7 m_fm_t1[4];
#pragma HLS ARRAY_PARTITION variable=m_fm_t1 complete dim=1
FIX_16_7 m_fm_t2[4];
#pragma HLS ARRAY_PARTITION variable=m_fm_t2 complete dim=1



//FIX_16_7 o_fm1[2];
FIX_FM_acc o_fm1[2];
#pragma HLS ARRAY_PARTITION variable=o_fm1 complete dim=1
//FIX_16_7 o_fm2[2];
FIX_FM_acc o_fm2[2];
#pragma HLS ARRAY_PARTITION variable=o_fm2 complete dim=1

int ch1;
int ch2;


FIX_WT bias_buf[16];
#pragma HLS array_partition variable=bias_buf dim=1 complete
FIX_WT W3x3[16][4][4];
#pragma HLS ARRAY_PARTITION variable=W3x3 complete dim=0

FIX_FM line_buffer2[16][16][2];
#pragma HLS ARRAY_PARTITION variable=line_buffer2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=line_buffer2 cyclic factor=2 dim=2
#pragma HLS resource variable=line_buffer core=RAM_2P_LUTRAM 
FIX_FM window_buffer[16][4][4];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

FIX_32_9 m_buffer_t[4][4];
#pragma HLS ARRAY_PARTITION variable=m_buffer_t complete dim=0

//FIX_FM out_tp_3x3[2][2];
FIX_FM out_tp_3x3[16][2][2];
#pragma HLS ARRAY_PARTITION variable=out_tp_3x3 complete dim=0

//FIX_FM DATA[16][4][4];
FIX_16_7 DATA[16][4][4];
#pragma HLS ARRAY_PARTITION variable=DATA complete dim=0
FIX_FM_acc ATOFM[4][2];
#pragma HLS ARRAY_PARTITION variable=ATOFM complete dim=0
FIX_FM_acc zero=0;
FIX_FM_acc six=6;
if(mode == 0)
{
	for(int ci=0; ci<16; ci+=2)
	{
		LOAD_W1x1_S16(weights, weight_buf1, weight_buf2, ci*4);
		for(int h=0; h<16; h++)
		{
			for(int w=0; w<16; w++)
			{
	#pragma HLS pipeline II=1
				C_BUF4_test(bottom1[ci  ][h][w], bottom1[ci+16][h][w], bottom2[ci  ][h][w], bottom2[ci+16][h][w], m_fm_t1);
				C_BUF4_test(bottom1[ci+1][h][w], bottom1[ci+17][h][w], bottom2[ci+1][h][w], bottom2[ci+17][h][w], m_fm_t2);
				for(int co=0; co<16; co++)
				{
		#pragma HLS UNROLL
					BTd_Multip4_Gg_AT(ATOFM[0],
							weight_buf1[co][0],m_fm_t1[0],
							weight_buf1[co][1],m_fm_t1[1],
							weight_buf1[co][2],m_fm_t1[2],
							weight_buf1[co][3],m_fm_t1[3]);
					BTd_Multip4_Gg_AT(ATOFM[1],
							weight_buf2[co][0],m_fm_t2[0],
							weight_buf2[co][1],m_fm_t2[1],
							weight_buf2[co][2],m_fm_t2[2],
							weight_buf2[co][3],m_fm_t2[3]);

					if(chout==0)
					{
						ch1=co;
						ch2=co+16;
					}
					else
					{
						ch1=co+32;
						ch2=co+48;
					}
					FIX_FM_acc res1 = top[ch1][h][w];
					FIX_FM_acc res2 = top[ch2][h][w];


					res1+=ATOFM[0][0]+ATOFM[1][0];
					res2+=ATOFM[0][1]+ATOFM[1][1];
					if(ci==14)
					{
						res1 = res1<0?zero:(res1>six?six:res1);
						res2 = res2<0?zero:(res2>six?six:res2);
					}

					top[ch1][h][w] = res1;
					top[ch2][h][w] = res2;

				}
			}
		}
	}
}
else
{
	for (int cho=0;  cho<4; cho++){
	LOAD_W3x3_S16_my(weights, W3x3,cho);
	load_bias_my(bias[cho], bias_buf);
	for (int w = 0; w < 16; w+=2)
	{
		for (int h = 0; h < 16; h+=2)
		{
#pragma HLS PIPELINE II=2
			for (int c = 0; c < 16; c++)
			{
#pragma HLS UNROLL
				
				FIX_FM read_in00;
				FIX_FM read_in01;

				FIX_FM read_in10;
				FIX_FM read_in11;
				if(cho==0)
				{
					read_in00 = bottom1[c][h][w];
					read_in10 = bottom1[c][h+1][w];

					read_in01 = bottom1[c][h][w+1];
					read_in11 = bottom1[c][h+1][w+1];
				}
				else if(cho==1)
				{
					read_in00 = bottom1[c+16][h][w];
					read_in10 = bottom1[c+16][h+1][w];

					read_in01 = bottom1[c+16][h][w+1];
					read_in11 = bottom1[c+16][h+1][w+1];
				}
				else if(cho==2)
				{
					read_in00 = bottom2[c][h][w];
					read_in10 = bottom2[c][h+1][w];

					read_in01 = bottom2[c][h][w+1];
					read_in11 = bottom2[c][h+1][w+1];
				}
				else
				{
					read_in00 = bottom2[c+16][h][w];
					read_in10 = bottom2[c+16][h+1][w];

					read_in01 = bottom2[c+16][h][w+1];
					read_in11 = bottom2[c+16][h+1][w+1];
				}
				window_buffer[c][3][3] = read_in11;
				window_buffer[c][3][2] = read_in10;
				window_buffer[c][3][1] = line_buffer2[c][h+1][1];
				window_buffer[c][3][0] = line_buffer2[c][h+1][0];	

				window_buffer[c][2][3] = read_in01;
				window_buffer[c][2][2] = read_in00;
				window_buffer[c][2][1] = line_buffer2[c][h][1];
				window_buffer[c][2][0] = line_buffer2[c][h][0];				
				
				line_buffer2[c][h][0] = read_in00;
				line_buffer2[c][h][1] = read_in01;

				line_buffer2[c][h+1][0] = read_in10;
				line_buffer2[c][h+1][1] = read_in11;


				if (w >= 2 )
				{
					ifm_trans_2row(window_buffer[c][2], window_buffer[c][3], DATA[c][2], DATA[c][3]);		
					if(h >= 2)
					{
						BTd_Multip4_Gg_AT(ATOFM[0],
								W3x3[c][0][0], DATA[c][0][0],
								W3x3[c][1][0], DATA[c][1][0],
								W3x3[c][2][0], DATA[c][2][0],
								W3x3[c][3][0], DATA[c][3][0]);
						BTd_Multip4_Gg_AT(ATOFM[1],
								W3x3[c][0][1], DATA[c][0][1],
								W3x3[c][1][1], DATA[c][1][1],
								W3x3[c][2][1], DATA[c][2][1],
								W3x3[c][3][1], DATA[c][3][1]);
						BTd_Multip4_Gg_AT(ATOFM[2],
								W3x3[c][0][2], DATA[c][0][2],
								W3x3[c][1][2], DATA[c][1][2],
								W3x3[c][2][2], DATA[c][2][2],
								W3x3[c][3][2], DATA[c][3][2]);
						BTd_Multip4_Gg_AT(ATOFM[3],
								W3x3[c][0][3], DATA[c][0][3],
								W3x3[c][1][3], DATA[c][1][3],
								W3x3[c][2][3], DATA[c][2][3],
								W3x3[c][3][3], DATA[c][3][3]);		
						A_OFM(ATOFM, out_tp_3x3[c], bias_buf[c]);
						if(cho==0)
						{
							top[c][h-1][w-1]=out_tp_3x3[c][0][0];
							top[c][h-1][w  ]=out_tp_3x3[c][0][1];
							top[c][h  ][w-1]=out_tp_3x3[c][1][0];
							top[c][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else if(cho==1)
						{
							top[c+16][h-1][w-1]=out_tp_3x3[c][0][0];
							top[c+16][h-1][w  ]=out_tp_3x3[c][0][1];
							top[c+16][h  ][w-1]=out_tp_3x3[c][1][0];
							top[c+16][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else if(cho==2)
						{
							top[c+32][h-1][w-1]=out_tp_3x3[c][0][0];
							top[c+32][h-1][w  ]=out_tp_3x3[c][0][1];
							top[c+32][h  ][w-1]=out_tp_3x3[c][1][0];
							top[c+32][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else
						{
							top[c+48][h-1][w-1]=out_tp_3x3[c][0][0];
							top[c+48][h-1][w  ]=out_tp_3x3[c][0][1];
							top[c+48][h  ][w-1]=out_tp_3x3[c][1][0];
							top[c+48][h  ][w  ]=out_tp_3x3[c][1][1];
						}
					}
					for (int r = 0; r < 4; r++)
					{
	#pragma HLS UNROLL
						DATA[c][0][r] = DATA[c][2][r];
						DATA[c][1][r] = DATA[c][3][r];
					}
				}

				for (int r = 0; r < 4; r++)
				{
#pragma HLS UNROLL
					window_buffer[c][0][r] = window_buffer[c][2][r];
					window_buffer[c][1][r] = window_buffer[c][3][r];
				}
			}
		}
	}
}
}
}



void share_1D2DWin_conv32_32(FIX_FM bottom1[32][16][16],
                          FIX_FM bottom2[32][16][16],
			  FIX_FM_acc top[2][32][16][16],
			  FIX_WT weights[16][64],
			  FIX_WT bias[4][16],
			  uint1 chout,
			  uint1 mode)
{

#pragma HLS array_partition variable=top dim=1 complete
#pragma HLS array_partition variable=top dim=2 complete
#pragma HLS array_partition variable=bottom1 dim=1 complete
#pragma HLS array_partition variable=bottom2 dim=1 complete

#pragma HLS array_partition variable=weights dim=1 complete

FIX_WT weight_buf1[16][4];
#pragma HLS array_partition variable=weight_buf1 dim=1 complete
#pragma HLS array_partition variable=weight_buf1 dim=2 complete
FIX_WT weight_buf2[16][4];
#pragma HLS array_partition variable=weight_buf2 dim=1 complete
#pragma HLS array_partition variable=weight_buf2 dim=2 complete


FIX_16_7 m_fm_t1[4];
#pragma HLS ARRAY_PARTITION variable=m_fm_t1 complete dim=1
FIX_16_7 m_fm_t2[4];
#pragma HLS ARRAY_PARTITION variable=m_fm_t2 complete dim=1


int ch1;
int ch2;


FIX_WT bias_buf[16];
#pragma HLS array_partition variable=bias_buf dim=1 complete
FIX_WT W3x3[16][4][4];
#pragma HLS ARRAY_PARTITION variable=W3x3 complete dim=0

FIX_FM line_buffer2[16][16][2];
#pragma HLS ARRAY_PARTITION variable=line_buffer2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=line_buffer2 cyclic factor=2 dim=2
#pragma HLS resource variable=line_buffer core=RAM_2P_LUTRAM 
FIX_FM window_buffer[16][4][4];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0


//FIX_FM out_tp_3x3[2][2];
FIX_FM out_tp_3x3[16][2][2];
#pragma HLS ARRAY_PARTITION variable=out_tp_3x3 complete dim=0

//FIX_FM DATA[16][4][4];
FIX_16_7 DATA[16][4][4];
#pragma HLS ARRAY_PARTITION variable=DATA complete dim=0
FIX_FM_acc ATOFM[16][4][2];
#pragma HLS ARRAY_PARTITION variable=ATOFM complete dim=0
FIX_FM_acc zero=0;
FIX_FM_acc six=6;
if(mode == 0)
{
	for(int ci=0; ci<16; ci+=2)
	{
		LOAD_W1x1_S16(weights, weight_buf1, weight_buf2, ci*4);
		for(int h=0; h<16; h++)
		{
			for(int w=0; w<16; w++)
			{
	#pragma HLS pipeline II=1
				C_BUF4_test(bottom1[ci  ][h][w], bottom1[ci+16][h][w], bottom2[ci  ][h][w], bottom2[ci+16][h][w], m_fm_t1);
				C_BUF4_test(bottom1[ci+1][h][w], bottom1[ci+17][h][w], bottom2[ci+1][h][w], bottom2[ci+17][h][w], m_fm_t2);
				for(int co=0; co<16; co++)
				{
		#pragma HLS UNROLL
					BTd_Multip4_Gg_AT(ATOFM[co][0],
							weight_buf1[co][0],m_fm_t1[0],
							weight_buf1[co][1],m_fm_t1[1],
							weight_buf1[co][2],m_fm_t1[2],
							weight_buf1[co][3],m_fm_t1[3]);
					BTd_Multip4_Gg_AT(ATOFM[co][1],
							weight_buf2[co][0],m_fm_t2[0],
							weight_buf2[co][1],m_fm_t2[1],
							weight_buf2[co][2],m_fm_t2[2],
							weight_buf2[co][3],m_fm_t2[3]);

					// BTd_Multip4_Gg_AT(ATOFM[0],
					// 		weight_buf1[co][0],bottom1[ci   ][h][w],
					// 		weight_buf1[co][1],bottom1[ci+16][h][w],
					// 		weight_buf1[co][2],bottom2[ci  ][h][w],
					// 		weight_buf1[co][3],bottom2[ci+16][h][w]);
					// BTd_Multip4_Gg_AT(ATOFM[1],
					// 		weight_buf2[co][0],bottom1[ci+1 ][h][w],
					// 		weight_buf2[co][1],bottom1[ci+17][h][w],
					// 		weight_buf2[co][2],bottom2[ci+1 ][h][w],
					// 		weight_buf2[co][3],bottom2[ci+17][h][w]);			
					ch1=co;
					ch2=co+16;
					FIX_FM_acc res1 = top[chout][ch1][h][w];
					FIX_FM_acc res2 = top[chout][ch2][h][w];


					res1+=ATOFM[co][0][0]+ATOFM[co][1][0];
					res2+=ATOFM[co][0][1]+ATOFM[co][1][1];
					if(ci==14)
					{
						res1 = res1<0?zero:(res1>six?six:res1);
						res2 = res2<0?zero:(res2>six?six:res2);
					}

					top[chout][ch1][h][w] = res1;
					top[chout][ch2][h][w] = res2;

				}
			}
		}
	}
}
else
{
	for (int cho=0;  cho<4; cho++){
	LOAD_W3x3_S16_my(weights, W3x3,cho);
	load_bias_my(bias[cho], bias_buf);
	for (int w = 0; w < 16; w+=2)
	{
		for (int h = 0; h < 16; h+=2)
		{
#pragma HLS PIPELINE II=2
			for (int c = 0; c < 16; c++)
			{
#pragma HLS UNROLL
				
				FIX_FM read_in00;
				FIX_FM read_in01;

				FIX_FM read_in10;
				FIX_FM read_in11;
				if(cho==0)
				{
					read_in00 = bottom1[c][h][w];
					read_in10 = bottom1[c][h+1][w];

					read_in01 = bottom1[c][h][w+1];
					read_in11 = bottom1[c][h+1][w+1];
				}
				else if(cho==1)
				{
					read_in00 = bottom1[c+16][h][w];
					read_in10 = bottom1[c+16][h+1][w];

					read_in01 = bottom1[c+16][h][w+1];
					read_in11 = bottom1[c+16][h+1][w+1];
				}
				else if(cho==2)
				{
					read_in00 = bottom2[c][h][w];
					read_in10 = bottom2[c][h+1][w];

					read_in01 = bottom2[c][h][w+1];
					read_in11 = bottom2[c][h+1][w+1];
				}
				else
				{
					read_in00 = bottom2[c+16][h][w];
					read_in10 = bottom2[c+16][h+1][w];

					read_in01 = bottom2[c+16][h][w+1];
					read_in11 = bottom2[c+16][h+1][w+1];
				}
				window_buffer[c][3][3] = read_in11;
				window_buffer[c][3][2] = read_in10;
				window_buffer[c][3][1] = line_buffer2[c][h+1][1];
				window_buffer[c][3][0] = line_buffer2[c][h+1][0];	

				window_buffer[c][2][3] = read_in01;
				window_buffer[c][2][2] = read_in00;
				window_buffer[c][2][1] = line_buffer2[c][h][1];
				window_buffer[c][2][0] = line_buffer2[c][h][0];				
				
				line_buffer2[c][h][0] = read_in00;
				line_buffer2[c][h][1] = read_in01;

				line_buffer2[c][h+1][0] = read_in10;
				line_buffer2[c][h+1][1] = read_in11;


				if (w >= 2 )
				{
					ifm_trans_2row(window_buffer[c][2], window_buffer[c][3], DATA[c][2], DATA[c][3]);		
					if(h >= 2)
					{
						BTd_Multip4_Gg_AT(ATOFM[c][0],
								W3x3[c][0][0], DATA[c][0][0],
								W3x3[c][1][0], DATA[c][1][0],
								W3x3[c][2][0], DATA[c][2][0],
								W3x3[c][3][0], DATA[c][3][0]);
						BTd_Multip4_Gg_AT(ATOFM[c][1],
								W3x3[c][0][1], DATA[c][0][1],
								W3x3[c][1][1], DATA[c][1][1],
								W3x3[c][2][1], DATA[c][2][1],
								W3x3[c][3][1], DATA[c][3][1]);
						BTd_Multip4_Gg_AT(ATOFM[c][2],
								W3x3[c][0][2], DATA[c][0][2],
								W3x3[c][1][2], DATA[c][1][2],
								W3x3[c][2][2], DATA[c][2][2],
								W3x3[c][3][2], DATA[c][3][2]);
						BTd_Multip4_Gg_AT(ATOFM[c][3],
								W3x3[c][0][3], DATA[c][0][3],
								W3x3[c][1][3], DATA[c][1][3],
								W3x3[c][2][3], DATA[c][2][3],
								W3x3[c][3][3], DATA[c][3][3]);		
						A_OFM(ATOFM[c], out_tp_3x3[c], bias_buf[c]);
						if(cho==0)
						{
							top[0][c][h-1][w-1]=out_tp_3x3[c][0][0];
							top[0][c][h-1][w  ]=out_tp_3x3[c][0][1];
							top[0][c][h  ][w-1]=out_tp_3x3[c][1][0];
							top[0][c][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else if(cho==1)
						{
							top[0][c+16][h-1][w-1]=out_tp_3x3[c][0][0];
							top[0][c+16][h-1][w  ]=out_tp_3x3[c][0][1];
							top[0][c+16][h  ][w-1]=out_tp_3x3[c][1][0];
							top[0][c+16][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else if(cho==2)
						{
							top[1][c][h-1][w-1]=out_tp_3x3[c][0][0];
							top[1][c][h-1][w  ]=out_tp_3x3[c][0][1];
							top[1][c][h  ][w-1]=out_tp_3x3[c][1][0];
							top[1][c][h  ][w  ]=out_tp_3x3[c][1][1];
						}
						else
						{
							top[1][c+16][h-1][w-1]=out_tp_3x3[c][0][0];
							top[1][c+16][h-1][w  ]=out_tp_3x3[c][0][1];
							top[1][c+16][h  ][w-1]=out_tp_3x3[c][1][0];
							top[1][c+16][h  ][w  ]=out_tp_3x3[c][1][1];
						}
					}
					for (int r = 0; r < 4; r++)
					{
	#pragma HLS UNROLL
						DATA[c][0][r] = DATA[c][2][r];
						DATA[c][1][r] = DATA[c][3][r];
					}
				}

				for (int r = 0; r < 4; r++)
				{
#pragma HLS UNROLL
					window_buffer[c][0][r] = window_buffer[c][2][r];
					window_buffer[c][1][r] = window_buffer[c][3][r];
				}
			}
		}
	}
}
}
}

