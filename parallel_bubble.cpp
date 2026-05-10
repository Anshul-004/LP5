#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// Parallel Bubble Sort
void parallel_bubble_sort(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n; i++) {

        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {

            if (arr[j] > arr[j + 1]) {

                #pragma omp critical
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {

    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements: ";

    for (int i = 0; i < n; i++)
        cin >> arr[i];

    parallel_bubble_sort(arr);

    cout << "Sorted Array: ";

    for (int x : arr)
        cout << x << " ";

    cout << endl;

    return 0;
}