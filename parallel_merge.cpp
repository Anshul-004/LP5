#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Merge Function
void merge(vector<int> &arr, int left, int mid, int right)
{
    vector<int> temp(right - left + 1);

    int i = left;
    int j = mid + 1;
    int k = 0;

    while (i <= mid && j <= right)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= right)
        temp[k++] = arr[j++];

    for (int i = left; i <= right; i++)
        arr[i] = temp[i - left];
}

// Sequential Merge Sort
void sequential_merge_sort(vector<int> &arr, int left, int right)
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    sequential_merge_sort(arr, left, mid);
    sequential_merge_sort(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

// Parallel Merge Sort
void parallel_merge_sort(vector<int> &arr, int left, int right, int depth = 0)
{
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;

    // Limit recursion depth to avoid too many threads
    if (depth <= 4)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                parallel_merge_sort(arr, left, mid, depth + 1);
            }

#pragma omp section
            {
                parallel_merge_sort(arr, mid + 1, right, depth + 1);
            }
        }
    }
    else
    {
        sequential_merge_sort(arr, left, mid);
        sequential_merge_sort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

void printArray(const vector<int> &arr)
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

    // Sequential Merge Sort Timing
    double start_seq = omp_get_wtime();

    sequential_merge_sort(seqArr, 0, n - 1);

    double end_seq = omp_get_wtime();

    // Parallel Merge Sort Timing
    double start_par = omp_get_wtime();

    parallel_merge_sort(parArr, 0, n - 1);

    double end_par = omp_get_wtime();

    // Output Sequential Result
    cout << "\nSequential Merge Sort Result:\n";
    printArray(seqArr);

    // Output Parallel Result
    cout << "\nParallel Merge Sort Result:\n";
    printArray(parArr);

    // Print Execution Times
    cout << "\nExecution Time:\n";
    cout << "Sequential Merge Sort Time: "
         << (end_seq - start_seq) << " seconds\n";

    cout << "Parallel Merge Sort Time: "
         << (end_par - start_par) << " seconds\n";

    return 0;
}
