/*
 * NaiveBayes test
 * 2016.10.31
 * author:soma62jp
 *                  */


#include "test.h"

//using namespace std;


NaiveBayes::NaiveBayes(const int &inum,const int &onum,const int &pnum):
	inputnum(inum),
	outputnum(onum),
	patternnum(pnum)
{
	// pattern*input
	Ii.resize(patternnum);				// ()内の数字が要素数になる
	for( int i=0; i<patternnum; i++ ){
		Ii[i].resize(inputnum);
	}

	Ti.resize(patternnum);
	Testi.resize(patternnum);

}

NaiveBayes::~NaiveBayes()
{
	// NOP
}


void  NaiveBayes::setInData(const int &pnum,const int &i,const double &value)
{
	if(pnum>=patternnum || i>=inputnum){
		//cout << "can't set Indata." << endl;
		outlog("can't set Indata.");
		return;
	}
	Ii[pnum][i] = value;
}

void NaiveBayes::setTeachData(const int &pnum,const double &value)
{
	if(pnum>=patternnum){
		//cout << "can't set Teachdata." << endl;
		outlog("can't set Teachdata.");
		return;
	}
	Ti[pnum] = value;
}

void NaiveBayes::setPredictData(const int &i,const double &value)
{
	if(i>=inputnum){
		//cout << "can't set Predictdata." << endl;
		outlog("can't set Predictdata.");
		return;
	}
	Testi[i] = value;
}



void NaiveBayes::train()
{
	gaussian_naive_bayes(Ii,Ti);
}

//void NaiveBayes::predict(const int &pnum)
void NaiveBayes::predict()
{
	std::vector<double> likehood = gaussian_naive_bayes_predict(Testi);

	for(int i=0;i<(int)likehood.size();i++){
		std::cout << likehood[i] << ",";	
	}
	std::cout << std::endl;	

}

void NaiveBayes::outlog(double value)
{
	//Form1->Memo1->Lines->Add(FloatToStr(value));
	std::cout << value << std::endl;
}

void NaiveBayes::outlog(std::string str)
{
	//Form1->Memo1->Lines->Add(str.c_str());
	std::cout << str << std::endl;
}

double NaiveBayes::gaussian(double &x, double &mean, double &variance) const
{
	return std::exp(-std::pow(x - mean, 2.) / (2. * variance)) / std::sqrt(variance);
}

void NaiveBayes::get_gaussian_params(const std::vector< std::vector<double> >& input, const std::vector<double>& target)
{

	double wrk;
	int cnt;
	int classnum;

	cnt=0;
	means.resize(outputnum);
	for(int i=0;i<outputnum;i++){
		for(int j=0;j<inputnum;j++){
			wrk=0;
			classnum=0;
			for(int k=0;k<patternnum;k++){
				if(Ti[k]==cnt){
					wrk+=Ii[k][j];
					classnum++;
				}
			}
			means[cnt].push_back(wrk / classnum);
		}
		classnums.push_back(classnum);
		cnt++;
	}

	cnt=0;
	variances.resize(outputnum);
	for(int i=0;i<outputnum;i++){
		for(int j=0;j<inputnum;j++){
			wrk=0;
			classnum=0;
			for(int k=0;k<patternnum;k++){
				if(Ti[k]==cnt){
					wrk+=(Ii[k][j]-means[cnt][j])*(Ii[k][j]-means[cnt][j]);
					classnum++;
				}
			}
			variances[cnt].push_back(wrk / classnum);
		}
		cnt++;
	}

}

void NaiveBayes::gaussian_naive_bayes(std::vector< std::vector<double> >& input, const std::vector<double>& target)
{
	//double mean,variance;
	//std::vector< std::vector<double> > likehood( patternnum , std::vector<double> (outputnum,0));

	get_gaussian_params(input, target);

#if 0
	for(int i=0;i<patternnum;i++){
		for(int j=0;j<outputnum;j++){
			for(int k=0;k<inputnum;k++){
				mean = means[j][k];
				variance = variances[j][k];
				//likehood[i][j]+=std::log(gaussian(Ii[i][k], mean, variance));
				likehood[i][j] += std::log( (double)classnums[j] / patternnum) + std::log(gaussian(input[i][k], mean, variance));
			}
		}
	}
#endif

#if 0
	for(int i=0;i<patternnum;i++){
		for(int j=0;j<outputnum;j++){
			std::cout << likehood[i][j] << ":";
		}
		std::cout << std::endl;
	}
#endif

}

std::vector<double> NaiveBayes::gaussian_naive_bayes_predict(std::vector<double>& target)
{
	double mean,variance;
	std::vector<double> likehood(outputnum,0);

	for(int j=0;j<outputnum;j++){
		for(int k=0;k<inputnum;k++){
			mean = means[j][k];
			variance = variances[j][k];
			//likehood[i][j]+=std::log(gaussian(Ii[i][k], mean, variance));
			likehood[j] += std::log( (double)classnums[j] / patternnum) + std::log(gaussian(target[k], mean, variance));
		}
	}

	return likehood;

}

int main()
{
  int i;
  double input[150][4];

  NaiveBayes nb(4,3,150);

  //ファイルの読み込み
	std::stringstream ss;
	std::ifstream ifs("iris.txt");
	if(!ifs){
        nb.outlog("--  入力エラー --");
        return 0;
    }

    //csvファイルを1行ずつ読み込む
	std::string str;
	int inputnum;
	int cnt = 0;
	while(getline(ifs,str)){
		std::string token;
		std::istringstream stream(str);

		//1行のうち、文字列とコンマを分割する
		inputnum = 0;
		while(getline(stream,token,',')){
			//すべて文字列として読み込まれるため
			//数値は変換が必要
			if(inputnum < 4){
				//double temp=stof(token); //stof(string str) : stringをfloatに変換

				// 文字列から数値に変換
				double temp;
				ss << token;
				ss >> temp;

				input[cnt][inputnum] = temp;

				ss.clear(); // 状態をクリア.
				ss.str(""); // 文字列をクリア.

				nb.setInData(cnt,inputnum,temp);
			}else{
				if(token=="s"){
					nb.setTeachData(cnt,0);
				}else if(token=="e"){
					nb.setTeachData(cnt,1);
				}else{
					nb.setTeachData(cnt,2);
				}
			}
			inputnum++;
		}
		cnt++;
	}

  nb.train();


  nb.outlog("--  Predict --");

  for(i=0;i<150;i++){
	nb.setPredictData(0,input[i][0]);
	nb.setPredictData(1,input[i][1]);
	nb.setPredictData(2,input[i][2]);
	nb.setPredictData(3,input[i][3]);
	nb.predict();
  }

  return 0;

}
