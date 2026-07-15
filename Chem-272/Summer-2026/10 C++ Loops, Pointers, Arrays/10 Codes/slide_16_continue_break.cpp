#include <iostream>

int main(void) {
  int i = 0;
  // Print only even numbers, up to 18
  for(int i = 0; i < 200; i++) {
    // if i is odd, then continue to the next iteration
    if(i % 2 == 1) {
      continue;
    }
    // print the number. This only gets run if i is even
    std::cout << i << std::endl;
    // if i is 18, break out of the loop
    if(i == 18) {
      break;
    }
  }
  std::cout << "Loop done!" << std::endl;
  return 0;
}