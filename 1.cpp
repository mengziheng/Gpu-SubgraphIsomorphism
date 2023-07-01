#include <iostream>
#include <cmath>
int main()
{
    int value = 8;
    float load_factor = 0.25;
    int len = (int)(value / load_factor);
    if (value / load_factor != (float)len)
        len++;
    printf("%d", len);
}