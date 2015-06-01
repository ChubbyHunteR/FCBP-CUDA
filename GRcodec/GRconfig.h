#ifndef CONFIG_H_
#define CONFIG_H_

/*
 * M is used when calculating the quotient and the remainder during GR coding and decoding
 */
#define M 4


/*
 * LOG_2_M is the log2 of M. It's made from a lookup table, but it's not really really really needed,
 * as we have complete control of M - it's not dynamic.
 */
#define LOG_2_M (_log_2_m[M]);
static int _log_2_m[] = {
		1, 1,
		2, 2,
		4, 4, 4, 4,
		8, 8, 8, 8, 8, 8, 8, 8
};

/*
 * Same as in ../config.h
 */
#define FILESET_SIZE 350

#endif /* CONFIG_H_ */
