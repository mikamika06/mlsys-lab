__global__ void wave_calc(float* out, int num_blocks, int num_sms, int blocks_per_sm) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int capacity = num_sms * blocks_per_sm;
        int num_waves = (num_blocks + capacity - 1) / capacity;   // ceiling division
        int remainder = num_blocks % capacity;
        int blocks_last = (remainder == 0) ? capacity : remainder;
        float last_util = (blocks_last * 1.0) / capacity;          // force real division
        out[0] = num_waves;
        out[1] = last_util;
    }
}
