GCC:= g++
FILE:= panorana_pictures_stitching.cpp

main:
	$(GCC) $(FILE) -o out -Wall	`pkg-config --cflags --libs opencv`

run:
	./out
