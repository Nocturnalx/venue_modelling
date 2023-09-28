#pragma once
struct point {
    int x;
    int y;
    int z;
};

struct dimensions {
    int xLength;
    int yLength;
    int zLength;
};

struct room{
    int pos_x;
    int pos_y;
    int pos_z;

    bool mirrored_x = 0;
    bool mirrored_y = 0;
    bool mirrored_z = 0;

    float totalAbs = 1;
};

struct coefs {
    float neg = 1;
    float pos = 1;
};