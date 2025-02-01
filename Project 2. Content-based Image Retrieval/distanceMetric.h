/*
  Xujia Qin 30th Jan, 2025
  S21
*/
#ifndef DISTANCE_METRIC_H
#define DISTANCE_METRIC_H

#include <vector>
#include <string>
using namespace std;


/*
 * Function that computes the sum_of_squared distances for all the images in the database,
 * and returns the distances as pair of vector with (filename, distance) in sorted order.

      Arg1: targetImageFeature: A vector that has the features of target Image.
      Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
      Arg3: filenames: A vector containing paths of all the image in database.

      return a sorted vector with (filename, distance) pairs.
 */
vector<pair<string, int>> sum_of_squared_difference(vector<float> &targetImageFeatures,vector<vector<float>> &featuresData, vector<char *> &filenames);

/*
 * Function that computes the histogram_intersection distance for all the images in the database,
 * and returns the distances as pair of vector with (filename, distance) in sorted order.

      Arg1: targetImageFeature: A vector that has the features of target Image.
      Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
      Arg3: filenames: A vector containing paths of all the image in database.

      returns a sorted vector with (filename, distance) pairs.
 */
vector<pair<string, float>> histogram_intersection(vector<float> &targetImageFeatures,
												   vector<vector<float>> &featuresData, vector<char *> &filenames);


// /*
//  * Function that computes the histogram_intersection distance(for multiple histogram features) for all the images in the database,
//  * and returns the distances as pair of vector with (filename, distance) in sorted order.

//       Arg1: targetImageFeature: A vector that has the features of target Image.
//       Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
//       Arg3: filenames: A vector containing paths of all the image in database.

//       returns a sorted vector with (filename, distance) pairs.
//  */
vector<pair<string, float>> histogram_intersection_for_2histograms(vector<float> &targetImageFeatures,
																   vector<vector<float>> &featuresData,
																   vector<char *> &filenames);

// /*
//  * Funtions that computes the entropy distance for all the images in the database with target Image.
//  * and returns the distances as a pair of vector with (filename, distance) in sorted order.

//         Arg1: targetImageFeature: A vector that has the features of target Image.
//         Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
//         Arg3: filenames: A vector containing paths of all the image in database.

//         returns a sorted vector with (filename, distance) pairs.
//  */
vector<pair<string, float>> entopyDistance(vector<float> &targetImageFeatures,
										   vector<vector<float>> &featuresData,
										   vector<char *> &filenames);


#endif // DISTANCE_METRIC_H