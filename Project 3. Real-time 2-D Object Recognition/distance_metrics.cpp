/*
  Xujia Qin 12th Feb, 2025
  S21
*/


#include <iostream>
#include "distance_metrics.h"
#include "filters.h"
#include "csv_utils.h"
#include<experimental/filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::experimental::filesystem;

// comparator function to sort vector pair based on value in ascending order.
bool cmp(pair<string, double> &a, pair<string, double> &b) {
  return a.second < b.second;
}

// comparator function to sort vector pair based on value in descending order.
bool cmp1(pair<string, double> &a, pair<string, double> &b) {
  return a.second > b.second;
}

// function to add label to the Image and display it.
int create_classified_image(cv::Mat &src,
							vector<pair<string, double>> &distances) {
  string label;
  label = distances[0].first;

  cv::Point text_position(80, 80);
  int font_size = 3;
  cv::Scalar font_color(0, 0, 0);
  int font_weight = 4;
  cv::putText(src, label, text_position, cv::FONT_HERSHEY_COMPLEX, font_size, font_color, font_weight);
  return 0;
}

// function to get standard-deviation for each-class.
vector<double> get_std(vector<vector<double>> &featureVectors) {
  vector<double> means;
  int col = 0;
  // compute means.
  while (col < 7) {
	double total_sum = 0;
	for (int i = 0; i < featureVectors.size(); i++) {
	  total_sum += featureVectors[i][col];
	}
	col += 1;
	means.push_back(total_sum/7);
  }

  // compute standard deviation
  vector<double> std;
  int col1 = 0;
  while (col1 < 7) {
	double var_sum = 0;
	for (int i = 0; i < featureVectors.size(); i++) {
	  var_sum += (featureVectors[i][col] - means[col1])*(featureVectors[i][col] - means[col1]);
	}
	col1 += 1;
	std.push_back(var_sum/7);
  }

  return std;
}

// get standard deviation for distances.
double get_std_distances(vector<pair<string, double>> &distances) {
  // calculate mean
  double mean = 0;
  for (int i = 0; i < distances.size(); i++) {
	mean += distances[i].second;
  }
  mean = mean/distances.size();

  // calculate std;
  double std = 0;
  for (int i = 0; i < distances.size(); i++) {
	std += (mean - distances[i].second)*(mean - distances[i].second);
  }
  std = std/distances.size();
  return std;
}

/*
 * A function that calculates chi-square distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg    : test Image RGB.
 * Args-3-testImg     : test Image thresholded.
 * Args-3-traindbpath : Path of the train database

 returns a Vector pair of distances with lable, distances as pairs in sorted order.
 */
vector<pair<string, double>> chisquareDistance(cv::Mat &testImg, char traindbPath[], char threstype[]) {
  vector<char *> filenames; // Vector to store filenames.
  vector<vector<double>> featureVectors; // Vector to store feature vectors.

  // get the feature vectors and associated labels.
  read_image_data_csv(traindbPath, filenames, featureVectors);
  vector<double> target_features = get_moments(testImg, threstype);
  // find the euclidean distance and store them in a vector<pairs>
  vector<pair<string, double>> distances;
  for (int i = 0; i < featureVectors.size(); i++) {
	double chisquare_dist = 0;
	for (int j = 0; j < featureVectors[i].size(); j++) {
	  double x1 = featureVectors[i][j];
	  double x2 = target_features[j];
	  chisquare_dist += ((x1 - x2)*(x1 - x2))/(x1 + x2);
	}
	distances.emplace_back(filenames[i], chisquare_dist/2);
  }
  // sort the distances.
  sort(distances.begin(), distances.end(), cmp);
  return distances;
}

/*
 * A function that calculates Manhattan distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg    : test Image RGB.
 * Args-3-testImg     : test Image thresholded.
 * Args-3-traindbpath : Path of the train database

 returns a Vector pair of distances with lable, distances as pairs in sorted order.
 */
vector<pair<string, double>> manhattanDistance(cv::Mat &testImg, char traindbPath[], char threshtype[]) {
  vector<char *> filenames; // Vector to store filenames.
  vector<vector<double>> featureVectors; // Vector to store feature vectors.

  // get the feature vectors and associated labels.
  read_image_data_csv(traindbPath, filenames, featureVectors);
  vector<double> target_features = get_moments(testImg, threshtype);
  // find the euclidean distance and store them in a vector<pairs>
  vector<pair<string, double>> distances;
  for (int i = 0; i < featureVectors.size(); i++) {
	double manhattan_dist = 0;
	for (int j = 0; j < featureVectors[i].size(); j++) {
	  double x1 = featureVectors[i][j];
	  double x2 = target_features[j];
	  manhattan_dist += abs(x1 - x2);
	}
	distances.emplace_back(filenames[i], manhattan_dist);
  }
  // sort the distances.
  sort(distances.begin(), distances.end(), cmp);
  return distances;
}

/*
 * A function that calculates scaled Euclidean distance for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg    : test Image RGB.
 * Args-3-testImg     : test Image thresholded.
 * Args-3-traindbpath : Path of the train database

 returns a Vector pair of distances with lable, distances as pairs in sorted order.
 */
vector<pair<string, double>> scaledEuclidean(cv::Mat &testImg, char traindbPath[], char threshtype[]) {
  vector<char *> filenames; // Vector to store filenames.
  vector<vector<double>> featureVectors; // Vector to store feature vectors.

  // get the feature vectors and associated labels.
  read_image_data_csv(traindbPath, filenames, featureVectors);
  vector<double> standard_deviations = get_std(featureVectors);
  cout << standard_deviations.size() << endl;
  vector<double> target_features = get_moments(testImg, threshtype);
  // find the euclidean distance and store them in a vector<pairs>
  vector<pair<string, double>> distances;
  for (int i = 0; i < featureVectors.size(); i++) {
	double euclidean_dist = 0;
	for (int j = 0; j < featureVectors[i].size(); j++) {
	  double x1 = featureVectors[i][j]/standard_deviations[j];
	  double x2 = target_features[j]/standard_deviations[j];
	  euclidean_dist += (x1 - x2)*(x1 - x2);
	}
	distances.emplace_back(filenames[i], sqrt(euclidean_dist));
  }
  // sort the distances.
  sort(distances.begin(), distances.end(), cmp);
  double std = get_std_distances(distances);
  if (distances[0].second > 3*std) {
	cout << "It is a new object" << endl;
	cout << "enter new label:" << endl;
	string label;
	cin >> label;
	collect_data(testImg, threshtype, label);
	::exit(-1);
  }

  return distances;
}

/*
 * A function that performs knn classification for the test-image with all
   the images in the database and returns the label of the image with least distance.
 * Args-1-colorImg        : test Image RGB.
 * Args-2-testImg         : test Image thresholded.
 * Args-3-traindbpath     : Path of the train database
 * Args-4-k value         : Number of nearest neighbours to consider.
 * Args-5-distance-metric : Type of distance metric to use.
 returns a vector pair with label,count pairs in sorted order(descending).
 */
vector<pair<string, double>> knnClassifier(cv::Mat &testImg,
										   char traindbpath[],
										   int k_value,
										   string distance_metric,
										   char threshtype[]) {
  vector<pair<string, double>> distances; // Vector pair to store euclidean-distances.
  if (distance_metric=="manhattan_dist")
	distances = manhattanDistance(testImg, traindbpath, threshtype);
  else if (distance_metric=="scaled_euclidean")
	distances = scaledEuclidean(testImg, traindbpath, threshtype);
  else if (distance_metric=="chi-square")
	distances = chisquareDistance(testImg, traindbpath, threshtype);

  unordered_map<string, double> counts; // Hashmap to store counts of first k-closest labels
  for (int i = 0; i < k_value; i++) {
	string key = distances[i].first;
	if (counts.find(key)==counts.end())
	  counts[key] = 1;
	else
	  counts[key] += 1;
  }

  unordered_map<string, double>::iterator itr; // iterating through the hash_map.
  vector<pair<string, double>> sorted_counts; // storing sorted hash_map as vector of pairs.
  for (itr = counts.begin(); itr!=counts.end(); itr++)
	sorted_counts.emplace_back(itr->first, itr->second);
  sort(sorted_counts.begin(), sorted_counts.end(), cmp1);
  return sorted_counts;
}

// function to get lablel.
string get_label(string path, char delimeter) {
  vector<string> words;
  size_t first;
  size_t last = 0;

  while ((first = path.find_first_not_of(delimeter, last))
	  !=string::npos) {
	last = path.find(delimeter, first);
	words.push_back(path.substr(first, last - first));
  }
  return words[6];
}

// function to perform evaluation.
int evaluation(fs::path path, string distance_metric, char threshtype[]) {
  unordered_map<string, double> umap = {
	  {{"Mouse", 0}, {"Stapler", 1}, {"HeadPhone", 2}, {"Clip", 3}, {"Phone", 4}, {"Scissors", 5}, {"EyeLiner", 6},
	   {"Comb", 7}, {"Cup", 8}, {"SunScreen", 9}}
  };

  for (const auto &entry : fs::directory_iterator(path)) {
	if (entry.is_directory()) {
	  // get the path and name of the subdirectory
	  fs::path subdir_path = entry.path();
	  // recursively call iterateDirectory for the subdirectory
	  evaluation(subdir_path, distance_metric, threshtype);
	} else {
	  if (entry.path().filename()!=".DS_Store")
		string img_path =
			entry.path();
	  cv::Mat color_image = cv::imread(entry.path()); // Mat object to store original frame.
	  if (color_image.empty()) {
		continue;
	  }
	  vector<pair<string, double>> counts;
	  string label = get_label(entry.path(), '/');
	  counts = knnClassifier(color_image,
							 "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/train.csv",
							 15,
							 distance_metric,
							 threshtype);
	  char original_label[256];
	  char filename[256] =
		  "/Users/jyothivishnuvardhankolla/Desktop/Project-3Real-time-object-2DRecognition/Project-3/eval.csv";
	  ::strcpy(original_label, label.c_str());
	  vector<double> feature;
	  feature.push_back(umap[counts[0].first]);
	  append_image_data_csv(filename, original_label, feature);
	}
  }
  return 0;
}