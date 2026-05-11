#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void sequential_bubble_sort(vector<int>& arr)
{
    int n = arr.size();

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort using Odd-Even Transposition
void parallel_bubble_sort(vector<int>& arr)
{
    int n = arr.size();

    for (int i = 0; i < n; i++)
    {
        // Even phase
#pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }

        // Odd phase
#pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Print Array
void printArray(const vector<int>& arr)
{
    for (int x : arr)
        cout << x << " ";

    cout << endl;
}

int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);

    cout << "Enter elements: ";

    for (int i = 0; i < n; i++)
        cin >> arr[i];

    // Create copies for both sorting methods
    vector<int> seqArr = arr;
    vector<int> parArr = arr;

    // Sequential Bubble Sort Timing
    double start_seq = omp_get_wtime();

    sequential_bubble_sort(seqArr);

    double end_seq = omp_get_wtime();

    // Parallel Bubble Sort Timing
    double start_par = omp_get_wtime();

    parallel_bubble_sort(parArr);

    double end_par = omp_get_wtime();

    // Output Sequential Result
    cout << "\nSequential Bubble Sort Result:\n";
    printArray(seqArr);

    // Output Parallel Result
    cout << "\nParallel Bubble Sort Result:\n";
    printArray(parArr);

    // Print Execution Times
    cout << "\nExecution Time:\n";

    cout << "Sequential Bubble Sort Time: "
         << (end_seq - start_seq) << " seconds\n";

    cout << "Parallel Bubble Sort Time: "
         << (end_par - start_par) << " seconds\n";

    return 0;
}
