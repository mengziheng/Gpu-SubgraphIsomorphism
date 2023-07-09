#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int main()
{
    vector<int> arr{5, 4, 3, 1};
    sort(arr.begin(), arr.end());
    for (int i = 0; i < arr.size(); i++)
        printf("%d ", arr[i]);
}