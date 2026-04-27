#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <ctime>
using namespace std;

// Bubble Sort
void bubble_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

// Parallel Sort
void parallel_sort(vector<int>& arr) {
    int n = arr.size();
    int mid = n / 2;

    vector<int> left(arr.begin(), arr.begin() + mid);
    vector<int> right(arr.begin() + mid, arr.end());

    thread t1(bubble_sort, ref(left));
    thread t2(bubble_sort, ref(right));

    t1.join();
    t2.join();

    // Merge manually (instead of sort for better compatibility)
    int i = 0, j = 0, k = 0;
    vector<int> result(n);

    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j])
            result[k++] = left[i++];
        else
            result[k++] = right[j++];
    }

    while (i < left.size()) result[k++] = left[i++];
    while (j < right.size()) result[k++] = right[j++];

    arr = result;
}

int main() {
    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter elements: ";
    for (int i = 0; i < n; i++) cin >> arr[i];

    cout << "Original: ";
    for (int x : arr) cout << x << " ";
    cout << endl;

    // Sequential
    vector<int> a1 = arr;
    clock_t start = clock();
    bubble_sort(a1);
    clock_t end = clock();

    cout << "Sequential: ";
    for (int x : a1) cout << x << " ";
    cout << "\nTime: " << (double)(end - start) / CLOCKS_PER_SEC << " sec\n";

    // Parallel
    vector<int> a2 = arr;
    start = clock();
    parallel_sort(a2);
    end = clock();

    cout << "Parallel: ";
    for (int x : a2) cout << x << " ";
    cout << "\nTime: " << (double)(end - start) / CLOCKS_PER_SEC << " sec\n";

    return 0;
}