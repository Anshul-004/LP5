#include <iostream>
#include <vector>
#include <thread>
using namespace std;

// ---------- Sequential Vector Addition ----------
vector<int> sequential_add(const vector<int>& A, const vector<int>& B) {
    vector<int> C(A.size());
    for (int i = 0; i < A.size(); i++)
        C[i] = A[i] + B[i];
    return C;
}

// ---------- Parallel Vector Addition ----------
vector<int> parallel_add(const vector<int>& A, const vector<int>& B) {
    int n = A.size();
    vector<int> C(n);

    int t = 2; // number of threads (simple for exam)
    int chunk = n / t;

    vector<thread> th;

    for (int i = 0; i < t; i++) {
        th.push_back(thread([&, i]() {
            int start = i * chunk;
            int end = (i == t - 1) ? n : start + chunk;

            for (int j = start; j < end; j++)
                C[j] = A[j] + B[j];   // same logic as add_pair
        }));
    }

    for (auto &x : th) x.join();

    return C;
}

// ---------- Main ----------
int main() {
    int N;
    cout << "Enter size of vectors: ";
    cin >> N;

    vector<int> A(N), B(N);

    cout << "Enter elements of vector A:\n";
    for (int &x : A) cin >> x;

    cout << "Enter elements of vector B:\n";
    for (int &x : B) cin >> x;

    // Sequential
    vector<int> C_seq = sequential_add(A, B);

    // Parallel
    vector<int> C_par = parallel_add(A, B);

    // Check results
    bool same = true;
    for (int i = 0; i < N; i++) {
        if (C_seq[i] != C_par[i]) {
            same = false;
            break;
        }
    }

    cout << "\nResults same: " << (same ? "Yes" : "No") << endl;

    cout << "Resultant Vector: ";
    for (int x : C_seq) cout << x << " ";

    return 0;
}