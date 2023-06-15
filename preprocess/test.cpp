#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <map>

using namespace std;

int main(int argc, char *argv[])
{
    int neighbour_numbers[1] = {10};
    int intersection_order[1] = {9};

    int intersection_order_length = 1;
    int min;
    int min_index;
    for (int i = 0; i < intersection_order_length; i++)
    {
        min = neighbour_numbers[i];
        min_index = i;
        for (int j = i + 1; j < intersection_order_length; j++)
        {
            if (neighbour_numbers[j] < min)
            {
                min = neighbour_numbers[j];
                min_index = j;
            }
        }

        int tmp = neighbour_numbers[i];
        neighbour_numbers[i] = neighbour_numbers[min_index];
        neighbour_numbers[min_index] = tmp;

        tmp = intersection_order[i];
        intersection_order[i] = intersection_order[min_index];
        intersection_order[min_index] = tmp;
    }

    for (int i = 0; i < intersection_order_length; i++)
        printf("%d ", neighbour_numbers[i]);
    printf("\n");
    for (int i = 0; i < intersection_order_length; i++)
        printf("%d ", intersection_order[i]);
    printf("\n");
}