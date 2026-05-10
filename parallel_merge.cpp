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

        if (arr[i] < arr[j])
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

// Parallel Merge Sort
void parallel_merge_sort(vector<int> &arr, int left, int right)
{

    if (left >= right)
        return;

    int mid = (left + right) / 2;

#pragma omp parallel sections
    {

#pragma omp section
        {
            parallel_merge_sort(arr, left, mid);
        }

#pragma omp section
        {
            parallel_merge_sort(arr, mid + 1, right);
        }
    }

    merge(arr, left, mid, right);
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

    parallel_merge_sort(arr, 0, n - 1);

    cout << "Sorted Array: ";

    for (int x : arr)
        cout << x << " ";

    cout << endl;

    return 0;
}