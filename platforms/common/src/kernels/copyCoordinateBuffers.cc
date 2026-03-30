KERNEL void copyFloatBuffer(GLOBAL float* RESTRICT source, GLOBAL float4* RESTRICT dest, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        dest[i].x = source[3*i];
        dest[i].y = source[3*i+1];
        dest[i].z = source[3*i+2];
    }
}

/**
 * Scatter a packed float[3*count] source into float4 dest at specified GPU slots.
 * The .w component of dest is preserved (charge or inverse mass).
 */
KERNEL void scatterFloatBuffer(GLOBAL const float* RESTRICT source, GLOBAL float4* RESTRICT dest,
                                GLOBAL const int* RESTRICT indices, int count) {
    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE) {
        int idx = indices[i];
        dest[idx].x = source[3*i];
        dest[idx].y = source[3*i+1];
        dest[idx].z = source[3*i+2];
    }
}

#ifdef SUPPORTS_DOUBLE_PRECISION
KERNEL void copyDoubleBuffer(GLOBAL double* RESTRICT source, GLOBAL double4* RESTRICT dest, int numAtoms) {
    for (int i = GLOBAL_ID; i < numAtoms; i += GLOBAL_SIZE) {
        dest[i].x = source[3*i];
        dest[i].y = source[3*i+1];
        dest[i].z = source[3*i+2];
    }
}

/**
 * Scatter a packed double[3*count] source into double4 dest at specified GPU slots.
 * The .w component of dest is preserved (charge or inverse mass).
 */
KERNEL void scatterDoubleBuffer(GLOBAL const double* RESTRICT source, GLOBAL double4* RESTRICT dest,
                                 GLOBAL const int* RESTRICT indices, int count) {
    for (int i = GLOBAL_ID; i < count; i += GLOBAL_SIZE) {
        int idx = indices[i];
        dest[idx].x = source[3*i];
        dest[idx].y = source[3*i+1];
        dest[idx].z = source[3*i+2];
    }
}
#endif