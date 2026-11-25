CFLAGS = -O2 -Wall -Wextra -std=c11 -ffp-contract=off
microqwen: host_main.c ../firmware/microqwen.c ../firmware/microqwen.h
	$(CC) $(CFLAGS) -I../firmware -o $@ host_main.c ../firmware/microqwen.c -lm
order: order_main.c ../firmware/microqwen.c ../firmware/microqwen.h
	$(CC) $(CFLAGS) -I../firmware -o $@ order_main.c ../firmware/microqwen.c -lm
clean:
	rm -f microqwen order
