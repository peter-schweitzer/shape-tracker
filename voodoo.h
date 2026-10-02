/**
 * This is voodoo.
 * The question is, is this too much.
 * [...]
 * Is this too much voodoo for the next ten centuries?
 *
 *  ~ Terry A. Davis
 */

#define _VA_CNT(type, ...) sizeof((type[]){__VA_ARGS__}) / sizeof(type)
