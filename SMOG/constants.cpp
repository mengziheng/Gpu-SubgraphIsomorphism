#include"constants.h"

int intersection_size = 6;
int restriction_size = 5;
int *intersection_orders = new int[6]{0, 0, 1, -1, 2, -1};
int *intersection_offset = new int[5]{0, 1, 3, 5, 6};
int *restriction = new int[5]{-1, 0, -1, -1, 3};
int *reuse = new int[5]{-1, -1, -1, 2, 3};
