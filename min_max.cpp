#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    int n;
    cout << "Enter count of numbers: ";
    cin >> n;

    vector<int> arr(n);
    cout << "Enter numbers: ";
    for (int &x : arr) cin >> x;

    long long sum = 0;
    int mn = INT_MAX, mx = INT_MIN;

    #pragma omp parallel for reduction(+:sum) reduction(min:mn) reduction(max:mx)
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }

    cout << "\nSum: " << sum;
    cout << "\nMin: " << mn;
    cout << "\nMax: " << mx;
    cout << "\nAverage: " << (double)sum / n;

    return 0;
}