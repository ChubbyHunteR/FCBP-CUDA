#ifndef CONFIG_H_
#define CONFIG_H_

/*
 * R defines the radius in number of pixels. Radius is number of pixels left, right and top from the "current" pixel taken into
 * account when calculating the average. All the taken pixels form an area of N pixels, equal to (R+1) times (2R+1) minus R.
 */
#define R 100
#define N (2 * R * (R  + 1) + 1)
#define THREADS 512
#define LOOP 20
#define MAX_PREDICTORS 10

#endif /* CONFIG_H_ */
