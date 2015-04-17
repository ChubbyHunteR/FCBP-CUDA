#ifndef CONFIG_H_
#define CONFIG_H_

/*
 * R defines the radius in number of pixels. Radius is number of pixels left, right and top from the "current" pixel taken into
 * account when searching for similar pixels (pixels with similar vector). All the taken pixels form an area of R_A pixels, equal to (R+1) times (2R+1) minus R.
 */
#define R 5
#define R_A (2 * R * (R  + 1))

/*
* D_R defines the radius of pixel's vector when comparing it to other pixels. In reality, we compare the vectors.
* D defines the vector's size. It equals to D_R * (D_R + 3)
*/
#define D_R 1
#define D (D_R * (D_R + 3))

/*
* M defines number of similar pixels that the prediction will be trained upon.
*/
#define M 7

/*
* THREADS defines the number of threads per block to run on the GPU.
*/
#define THREADS 1024

#define LOOP 20

#endif /* CONFIG_H_ */
