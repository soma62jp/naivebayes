//---------------------------------------------------------------------------

#ifndef NaiveBayesH
#define NaiveBayesH

#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <ctime>        // time
#include <cstdlib>      // srand,rand
#include <fstream>
#include <string>
#include <sstream> //文字ストリーム

//---------------------------------------------------------------------------
//using namespace std;

class NaiveBayes
{

	public:
		NaiveBayes(const int &inum,const int &onum,const int &pnum);
		~NaiveBayes();
		void setInData(const int &pnum,const int &i,const double &value);
		void setTeachData(const int &pnum,const double &value);
		void setPredictData(const int &i,const double &value);
		void train();
		void predict();

		void outlog(std::string str);
		void outlog(double value);

	private:
		std::vector< std::vector<double> > Ii;				// 入力データ
		std::vector<double> Ti;								// 教師信号
		std::vector<double> Testi;			// テストデータ

		template <typename T> std::string tostr(const T& t)
		{
			std::ostringstream os; os<<t; return os.str();
		}

		double gaussian(double &x, double &mean, double &variance) const;
		void get_gaussian_params(const std::vector< std::vector<double> >& input, const std::vector<double>& target);
		void gaussian_naive_bayes(std::vector< std::vector<double> >& input, const std::vector<double>& target);
		std::vector<double> gaussian_naive_bayes_predict(std::vector<double>& target);

		const int inputnum;
		const int outputnum;
		const int patternnum;

		std::vector< std::vector<double> > means; 				// outputnum*inputnum
  		std::vector< std::vector<double> > variances;			// outputnum*inputnum
		std::vector<int> classnums;



};

#endif
