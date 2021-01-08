// Liam Howes 5880331

#include <iostream>
#include <cmath>
#include <cstdlib> // for getting random
#include <omp.h>

#include <string> // getting data from file
// using "time ./test" in sandcastle shell in conjunction with clock_t to get time for JUST KNN (parallel part) execution
#include <iomanip> // need this to use setprecision() when displaying an accurate execution time in seconds
using namespace std;

#define k 3 // define our k value
#define num_cols 8 // doubled dimensionality of 4D
#define num_rows 10000 // size of our data array
const int choice  = 1; // 0 = regression, 1 = classification
const int thread_count = 8;
const int split = 1250;//num_rows/thread_count



double distance_function(double point1[num_cols-1], double point2[num_cols-1]){
    // need to get absolute value, in order to avoid negative distances
    double result = pow((point1[0] - point2[0]), 2);
    for(int i=1; i<num_cols-1; i++){
        result = result - pow((point1[i] - point2[i]), 2);
    }
    return sqrt(abs(result));
}

// mean is used for the regression problem (finds a real value)
double mean(double closest_data[k][num_cols+1]){
    double sum = 0;
    for(int i=0; i<k; i++){
        sum += closest_data[i][num_cols-1]; 
    }
    return (double)sum/k; // average of closest weights/smartphone ownership
}

// mode is used for the classification problem (finds a discrete value based on "votes")
int mode(double closest_data[k][num_cols+1]){
    int zero_count = 0;
    int one_count = 0;
    for(int i=0; i<k; i++){
        if(closest_data[i][num_cols-1]==0){
            zero_count += 1;
        }
        else one_count += 1;
    }
    if(zero_count>one_count){
        return 0;
    }
    else return 1;
}

void KNN(double data[split][num_cols+1], double query[num_cols-1], double  nearest_neighbours[k][num_cols+1]){
    for(int i=0; i<split; i++){ //for every row ...
        double value[num_cols-1];
        for(int j = 0; j<num_cols-1; j++){//and column in search data
        // store the num_cols-1 query variables
            value[j] = data[i][j];
        }
        double distance = distance_function(value, query); //distance from current height/weight and query height/weight
            // store the height, weight index and the distance in an array
        
        data[i][num_cols] = distance;
    }
    // sort the array from smallest distance, to largest. Bubble Sort.
    double tmp;
    for(int i=0; i<(split-1); i++){
        for(int j=i+1; j<split; j++){
            // if the distance at j is less than the distance at i
            if(data[j][num_cols] < data[i][num_cols]){
                // then swap their places
                for(int t=0; t<num_cols+1; t++){
                    tmp = data[i][t];
                    data[i][t] = data[j][t];
                    data[j][t] = tmp;
                }
            }
        }
    }
    
    // find k-nearest neighbours
    for(int i=0; i<k; i++){
        for(int b=0; b<num_cols+1; b++){ // data is now sorted in ascending distance order, so just grab the first k elements
            nearest_neighbours[i][b] = data[i][b];
        }
    }
}

int main(){
    srand(time(NULL)); // random seed
    // Regression data:
    // A regression problem for the KNN algorithm gives a real number value as the output.
    // Eg. use data of people's heights and weights to guess the weight of someone with a given height. 

    // Classifcation data:
    // A classification problem for the KNN algorithm gives a discrete value as the output.
    // Eg. use data to determine, based on a person's age if they have a smartphone or not (yes or no).

    cout<<"Number of rows of data = "<<num_rows<<endl;

//regression
    if(choice==0){
        double reg_data[num_rows][num_cols]; //8D array for higher dimensional tests

        // fill arrray with training data: random numbers between 1 and 1000
        double upperbound = 30;
        double lowerbound = 1;
        for(int i=0; i<num_rows; i++){
            for(int j=0; j<num_cols; j++){
                reg_data[i][j] = (upperbound-(lowerbound))*((double)rand()/(double)RAND_MAX) + (lowerbound); // random step between 1 and 30
            }
        }

        double reg_query[num_cols-1]; // query has 7 values
        for(int i=0; i<num_cols-1; i++){
            reg_query[i] = (upperbound-(lowerbound))*((double)rand()/(double)RAND_MAX) + (lowerbound); // random step between 1 and 30
        }

        cout<<"regression model.\n";
        clock_t start, end;
        start = clock();
        double final_sorted_data[k*thread_count][num_cols+1]; //+1 to store the distance value for each training data (stored in last column)
        #pragma omp parallel num_threads(thread_count)
        {   
            // split the data array into sections for each thread
            int thread = omp_get_thread_num();
            double data_partition[split][num_cols+1];
            double local_query[num_cols-1];
            int p=-1;
            for(int i=(thread)*(split); i<split*(thread+1); i++){
                p++;
                for(int j=0; j<num_cols; j++){
                    data_partition[p][j] = reg_data[i][j];
                }
            }
            for(int i=0; i<num_cols-1; i++){
                local_query[i] = reg_query[i];
            }
            // each thread gets a partition of the full data
            double nearest_neighbours[k][num_cols+1]; //+1 to store the distance value for each training data (stored in last column)
            KNN(data_partition, local_query, nearest_neighbours);
            // store each threads nearest neighbours in a combined, final array
            int q=-1;
            for(int i=(thread)*(k); i<k*(thread+1); i++){
                q++;
                for(int j=0; j<num_cols+1; j++){
                    #pragma omp critical // writing to shared array
                    final_sorted_data[i][j] = nearest_neighbours[q][j];
                }
            }
        }
        // sort the combined, master array
        double tmp;
        for(int i=0; i<((k*thread_count)-1); i++){
            for(int j=i+1; j<(k*thread_count); j++){
                // if the distance at j is less than the distance at i
                if(final_sorted_data[j][num_cols] < final_sorted_data[i][num_cols]){
                    // then swap their places
                    for(int t=0; t<num_cols+1; t++){
                        tmp = final_sorted_data[i][t];
                        final_sorted_data[i][t] = final_sorted_data[j][t];
                        final_sorted_data[j][t] = tmp;
                    }
                }
            }
        }
        // get k closest neighbours
        double closest_data[k][num_cols+1];
        for(int i=0; i<k; i++){
            for(int b=0; b<(num_cols+1); b++){
                closest_data[i][b] = final_sorted_data[i][b]; 
            }
        }
        
        double answer = mean(closest_data); // MEAN is used for regression problem
        std::cerr<<"estimated 8th column value = "<<answer<<std::endl;

        end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        std::cout<<"execution time: ";
        std::cout<<std::fixed;
        std::cout<< std::setprecision(9) <<time_taken<<" seconds\n";   
    }
//classification
    else if(choice==1){
        double clas_data[num_rows][num_cols]; //8D array for higher dimensional tests

        // fill arrray with training data: random numbers between 1 and 1000
        double upperbound = 30;
        double lowerbound = 1;
        for(int i=0; i<num_rows; i++){
            for(int j=0; j<num_cols-1; j++){
                clas_data[i][j] = (upperbound-(lowerbound))*((double)rand()/(double)RAND_MAX) + (lowerbound); // random step between 1 and 30
            }
            clas_data[i][num_cols-1] = rand() % 2; // randomly 0 or 1 (last column always has to be 0 or 1, binary choice for simple classifcation problem)

        }

        double clas_query[num_cols-1]; // query has 7 values
        for(int i=0; i<num_cols-1; i++){
            clas_query[i] = (upperbound-(lowerbound))*((double)rand()/(double)RAND_MAX) + (lowerbound); // random step between 1 and 30
        }

        cout<<"classification model.\n";
        clock_t start, end;
        start = clock();
        double final_sorted_data[k*thread_count][num_cols+1]; //+1 to store the distance value for each training data (stored in last column)
        #pragma omp parallel num_threads(thread_count)
        {   
            // split the data array into sections for each thread
            int thread = omp_get_thread_num();
            double data_partition[split][num_cols+1];
            double local_query[num_cols-1];
            int p=-1;
            for(int i=(thread)*(split); i<split*(thread+1); i++){
                p++;
                for(int j=0; j<num_cols; j++){
                    data_partition[p][j] = clas_data[i][j];
                }
            }
            for(int i=0; i<num_cols-1; i++){
                local_query[i] = clas_query[i];
            }
            // each thread gets a partition of the full data
            double nearest_neighbours[k][num_cols+1]; //+1 to store the distance value for each training data (stored in last column)
            KNN(data_partition, local_query, nearest_neighbours);
            // store each threads nearest neighbours in a combined, final array
            int q=-1;
            for(int i=(thread)*(k); i<k*(thread+1); i++){
                q++;
                for(int j=0; j<num_cols+1; j++){
                    #pragma omp critical // writing to shared array
                    final_sorted_data[i][j] = nearest_neighbours[q][j];
                }
            }
        }
        // sort the combined, master array
        double tmp;
        for(int i=0; i<((k*thread_count)-1); i++){
            for(int j=i+1; j<(k*thread_count); j++){
                // if the distance at j is less than the distance at i
                if(final_sorted_data[j][num_cols] < final_sorted_data[i][num_cols]){
                    // then swap their places
                    for(int t=0; t<num_cols+1; t++){
                        tmp = final_sorted_data[i][t];
                        final_sorted_data[i][t] = final_sorted_data[j][t];
                        final_sorted_data[j][t] = tmp;
                    }
                }
            }
        }
        // get k closest neighbours
        double closest_data[k][num_cols+1];
        for(int i=0; i<k; i++){
            for(int b=0; b<(num_cols+1); b++){
                closest_data[i][b] = final_sorted_data[i][b]; 
            }
        }
        
        double answer = mode(closest_data); // MODE is used for classification problem
        std::cerr<<"estimated 8th column value = "<<answer<<std::endl;

        end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        std::cout<<"execution time: ";
        std::cout<<std::fixed;
        std::cout<< std::setprecision(9) <<time_taken<<" seconds\n";   
    }
    
    return 0;
}