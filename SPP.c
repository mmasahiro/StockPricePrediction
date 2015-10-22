#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define N 20
//#define n 5
#define T 1 
#define Theta 0.0 
#define Gamma 1
#define A 1
#define B 1
#define Alpha 0.3
#define Mu 0.5
#define SEED 151015

#define n_I 20
#define n_H 40
#define n_O 10

void global(char *fileName);
void global_test(char *fileName, double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]);
void initialize_weight(double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]);
void generate_S(int m, double S[], char fileName[]);
void generate_test_S(double S[], char fileName[]);
void generate_Global_x(double gx[], double S[]);
void generate_Local_x(double lx[], double S[]);
double sigmoid_func(double z);
double error_func(double x[], double x3[]);
void BP_global(double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[], double x[], double x1[], double x2[], double x3[]);
void NN_global(double x[], double x3[], double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]);
void NN_global_test(double x[], double x3[], double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]);
void Synthsizing(double Z[], double gZ[], double lZ[]);


int main(void){
  char fileName[] = "data_panasonic_practice1.dat";
  double S[N+1], gx[N+1], lx[N+1], x[N+1];
  double x3[n_O], w1[n_H][n_I], w2[n_O][n_H];
  double x1_bias[n_H], x2_bias[n_O];
  int i,j,k;

  srand48(SEED);

  global(fileName);
  //local(fileName);

  Synthsizing(x3, gx, lx);

  return 0;
}


void global(char *fileName){
  double S[N+1], gx[N+1];
  double x3[n_O], w1[n_H][n_I], w2[n_O][n_H];
  double x1_bias[n_H], x2_bias[n_O];
  int i;

  printf("global feature\n");

  for(i=0; i<n_O; i++){
    x3[i] = 0.0;
  }

  //各重みの初期化
  initialize_weight(w1, w2, x1_bias, x2_bias);

  for(i=0; i<10000; i++){
    //データの取得
    generate_S(i, S, fileName);
    generate_Global_x(gx, S);

    //printf("-------------------------\n");
    //printf("%d ",i);
    NN_global(gx, x3, w1, w2, x1_bias, x2_bias);
    printf("E(%d) = %lf\n",i , error_func(gx,x3));
    //sleep(1);
  }

  //以下テスト

  global_test(fileName, w1, w2, x1_bias, x2_bias);

  return;
}
/*
void local(char *fileName){
  double S[N+1], lx[N+1], x[N+1];
  double x3[n_O], w1[n_H][n_I], w2[n_O][n_H];
  double x1_bias[n_H], x2_bias[n_O];
  int i;

  printf("local feature\n");

  for(i=0; i<n_O; i++){
    x3[i] = 0.0;
  }

  //各重みの初期化
  initialize_weight(w1, w2, x1_bias, x2_bias);

  for(i=0; i<100; i++){
    //データの取得
    generate_S(i, S, fileName);
    generate_Local_x(lx, S);

    //printf("-------------------------\n");
    //printf("%d ",i);
    NN_local(lx, x3, w1, w2, x1_bias, x2_bias);
    printf("  (%d)\n",i);
    //sleep(1);
  }


  return;
}
*/


//----------

void global_test(char *fileName, double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]){
  double S[N+1], gx[N+1] ,x3[n_O];
  int i;

  generate_test_S(S, fileName);
  //generate_S(15, S, fileName);
  for(i=0; i<=N; i++){
    //printf("S(%d) = %lf\n",i ,S[i]);
  }

  generate_Global_x(gx, S);

  for(i=0; i<N; i++){
    //printf("gx(%d) = %lf\n",i ,gx[i]);
  }

  NN_global_test(gx, x3, w1, w2, x1_bias, x2_bias);

  for(i=0; i<n_O; i++){
    printf("x3[%d] = %lf  |  gx[%d] = %lf\n", i, x3[i], i+(n_I-n_O), gx[i+(n_I-n_O)]);
  }

  return;
}


//----------


void initialize_weight(double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]){
  int i,j,k;

  //入力層と中間層の間の重みの初期化
  for(j=0; j<n_H; j++){
    for(k=0; k<n_I; k++){
      w1[j][k] = drand48()*0.02 - 0.01;
      //printf("[%d][%d] %lf\n", j, k, w1[j][k]);
    }
  }

  //中間層と出力層の間の重みの初期化
  for(i=0; i<n_O; i++){
    for(j=0; j<n_H; j++){
      w2[i][j] = drand48()*0.02 - 0.01;
      //printf("[%d][%d] %lf\n", i, j, w2[i][j]);
    }
  }

  //しきい値の初期化
  for(j=0; j<n_H; j++){
    x1_bias[j] = drand48()*0.02 - 0.01;
  }

  for(i=0; i<n_O; i++){
    x2_bias[i] = drand48()*0.02 - 0.01;
  }

  return;
}

void generate_S(int m, double S[], char fileName[]){
  int i;
  FILE *fp;
  double a,b,c,d,e,f;
  char s[256];
  double sum = 0.0;

  fp = fopen(fileName, "r");
  if(fp == NULL){
    printf("file open error.\n");
    exit(-1);
  }
  //日付 始値 高値 安値 終値 出来高 調整後終値*

  //余分なデータの削除
  for(i=0; i<m%(65-n_I+1); i++){
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf", s, &a, &b, &c, &d, &e, &f);
  }

  //データの取得
  for(i=1; i<=N; i++){
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf", s, &S[i], &a, &b, &c, &d, &e);
  }

  //データの平均値の取得
  for(i=1; i<=N; i++){
    sum += S[i];
  }
  S[0] = sum/N;

  for(i=0; i<=N; i++){
    //printf("S[%d] = %lf\n", i, S[i]);
  }
  fclose(fp);
  return;
}

void generate_test_S(double S[], char fileName[]){
  int i;
  FILE *fp;
  double a,b,c,d,e,f;
  char s[256];
  double sum = 0.0;

  fp = fopen(fileName, "r");
  if(fp == NULL){
    printf("file open error.\n");
    exit(-1);
  }
  //日付 始値 高値 安値 終値 出来高 調整後終値*

  //余分なデータの削除
  for(i=0; i<55; i++){
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf", s, &a, &b, &c, &d, &e, &f);
  }

  //データの取得
  for(i=1; i<=N; i++){
    fscanf(fp, "%s %lf %lf %lf %lf %lf %lf", s, &S[i], &a, &b, &c, &d, &e);
  }

  //データの平均値の取得
  for(i=1; i<=(n_I - n_O); i++){
    sum += S[i];
  }
  S[0] = sum/(n_I - n_O);

  for(i=0; i<=N; i++){
    //printf("S[%d] = %lf\n", i, S[i]);
  }
  fclose(fp);
  return;
}

void generate_Global_x(double gx[], double S[]){
  int i;

  //全体的特徴の抽出
  for(i=0; i<N; i++){
    gx[i] = 1.0 / 2.0 * (1 + tanh((S[i+1] - S[0]) / A) );
    //printf("[%d] gx = %lf\n", i, gx[i]);
  }
  return;
}

void generate_Local_x(double lx[], double S[]){
  int i;
  double dS[N+1], sum=0.0;

  //偏差を求める
  for(i=1; i<N; i++){
    dS[i] = S[i] - S[i+1];
    sum = sum + dS[i];
  }
  dS[0] = sum / (N - 1);

  //部分的特徴の抽出
  for(i=0; i<(N-1); i++){
    lx[i] = 1.0 / 2.0 * (1 + tanh((dS[i+1] - dS[0]) / B) );
    //printf("[%d] lx = %lf \n", i, lx[i]);
  }

  return;
}

//----------

double sigmoid_func(double z){
  //シグモイド関数
  return 1 / (1 + exp( -(z - Theta) / T));
}

double error_func(double x[], double x3[]){
  int i;
  double sum = 0.0;

  //平均二乗誤差
  for(i=n_I - n_O; i<n_I; i++){
    sum = sum + pow((/*sigmoid_func*/(x[i]) - x3[i-(n_I - n_O)]), 2);
  }
  return sum;
}

void BP_global(double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[], double x[], double x1[], double x2[], double x3[]){
  double sum = 0.0;
  double dw1[n_H][n_I], dw2[n_O][n_H], dx1_bias[n_H], dx2_bias[n_O];
  double r[n_O], r_hat[n_H];
  int i,j,k,l;

  //中間層と出力層の間の重みの変化量を計算
  for(i=0; i<n_O; i++){
    r[i] = (/*sigmoid_func*/(x[i+(n_I - n_O)]) - x3[i]) * x3[i] * (1 - x3[i]);

    //printf("(%d)|%lf - %lf| = %lf\n", i+(n_I - n_O), /*sigmoid_func*/(x[i+(n_I - n_O)]), x3[i], fabs(/*sigmoid_func*/(x[i+(n_I - n_O)]) - x3[i]));

    for(j=0; j<n_H; j++){
      dw2[i][j] = Mu * r[i] * x2[j];
      //printf("(%d, %d) [r:%lf [dw2:%lf\n", j, i, r[i], dw2[i][j]);
    }
  }

  for(i=0; i<n_O; i++){
    dx2_bias[i] = Mu * r[i] * 1;
  }

  
  //入力層と中間層の間の重みの変化量を計算
  for(j=0; j<n_H; j++){
    sum = 0.0;
    for(i=0; i<n_O; i++){
      sum = sum + w2[i][j] * r[i];
    }

    r_hat[j] = sum * x2[j] * (1 - x2[j]);
    for(k=0; k<(n_I - 1); k++){
      dw1[j][k] = Mu * r_hat[j] * x1[k];
      //print("(%d, %d) [r:%lf [dw2:%lf\n", k, j, r_hat[j], dw1[j][k]);
    }
  }

  for(j=0; j<n_H; j++){
    dx1_bias[j] = Mu * r[j] * 1;
  }


  //各重みの値を変える
  for(j=0; j<n_H; j++){
    for(i=0; i<n_O; i++){
      //printf("w2(%d,%d) [%lf ->", j, i, w2[i][j]);
      w2[i][j] = w2[i][j] + dw2[i][j];
      //printf(" %lf]\n", w2[i][j]);
    }
  }

  for(i=0; i<n_O; i++){
    x2_bias[i] = x2_bias[i] + dx2_bias[i];
  }

  for(j=0; j<n_H; j++){
    for(k=0; k<(n_I - 1); k++){
      //printf("w1 [%lf ->", w1[j][k]);
      w1[j][k] = w1[j][k] + dw1[j][k];
      //printf("%lf]\n", w1[j][k]);
    }
  }

  for(j=0; j<n_H; j++){
    x1_bias[j] = x1_bias[j] + dx1_bias[j];
  }

  return;
}

void NN_global(double x[], double x3[], double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]){
  double x1[n_I], x2[n_H];
  double sum = 0.0;
  int i,j,k,l;

  for(k=0; k<(n_I - 1); k++){
    x1[k] = x[k];

    if(k > n_O){
      //printf("(x1[%d] = %lf) - (x3[%d] = %lf) = (%lf)\n", k, sigmoid_func(x1[k]), k-(n_O - 1), x3[k-(n_O - 1)], fabs(sigmoid_func(x1[k]) - x3[k-(n_O - 1)-1]));
    }else{
      //printf("x1[%d] = %lf\n", k, sigmoid_func(x1[k]));
    }
  }

  //データからと出力層からの値の差の絶対値をとる
  for(i=(n_I - 1) - (n_O - 1); i<(n_I - 1); i++){//*
    //printf("--[%d] [%d]\n", i, i-((n_I - 1) - (n_O - 1)));

    x1[i] = fabs(x1[i] - x3[i-((n_I - 1) - n_O)]) * Gamma;
  }

  //入力層の素子の値を計算する
  for(k=0; k<(n_I - 1); k++){
    //x1[k] = sigmoid_func(x1[k]);
    if(k >= n_O){
      //printf("x1[%d] = %lf\n", k, x1[k]);
    }else{
      //printf("x1[%d] = %lf\n", k, x1[k]);
    }
  }

  //printf("-------\n");

  //中間層の素子の値を計算する
  for(j=0; j<n_H; j++){
    sum = 0.0;
    for(k=0; k<(n_I - 1); k++){
      sum = sum + w1[j][k] * x1[k];
    }
    x2[j] = sigmoid_func(sum + x1_bias[j]);
    //printf("x2[%d] = %lf\n", j, x2[j]);
    //printf("%lf - %lf\n", sigmoid_func(sum), sigmoid_func(sum + x1_bias[j]));
  }

  //printf("-------\n");

  //出力層の素子の値を計算する
  for(i=0; i<n_O; i++){
    sum = 0.0;
    for(j=0; j<n_H; j++){
      sum = sum + w2[i][j] * x2[j];
    }
    x3[i] = sigmoid_func(sum + x2_bias[i]);
    //printf("x3[%d] = %lf\n", i+n_O, x3[i]);
  }

  //学習させる
  BP_global(w1, w2, x1_bias, x2_bias, x, x1, x2, x3);


  return;
}

//----------

void NN_global_test(double x[], double x3[], double w1[n_H][n_I], double w2[n_O][n_H], double x1_bias[], double x2_bias[]){
  double x1[n_I], x2[n_H];
  double sum = 0.0;
  int i,j,k,l;

  for(k=0; k<(n_I - 1); k++){
    x1[k] = x[k];

    if(k >= (n_I - n_O)){
      x1[k] = 0.0;
    }
  }

  //中間層の素子の値を計算する
  for(j=0; j<n_H; j++){
    sum = 0.0;
    for(k=0; k<(n_I - 1); k++){
      sum = sum + w1[j][k] * x1[k];
    }
    x2[j] = sigmoid_func(sum + x1_bias[j]);
  }

  //出力層の素子の値を計算する
  for(i=0; i<n_O; i++){
    sum = 0.0;
    for(j=0; j<n_H; j++){
      sum = sum + w2[i][j] * x2[j];
    }
    x3[i] = sigmoid_func(sum + x2_bias[i]);
  }

  return;
}

//----------
void Synthsizing(double Z[], double gZ[], double lZ[]){
  int i;

  //合成する
  for(i=(n_I - 1); i<N; i++){
    Z[i] = Alpha * gZ[i] + (1 - Alpha) * lZ[i];
  }
  return;
}
