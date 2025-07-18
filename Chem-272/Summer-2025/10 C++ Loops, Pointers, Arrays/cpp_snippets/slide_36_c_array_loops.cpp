#include <iostream>

int main(void) {
  int arr[10] = { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21 };

  for(int i = 0; i < 10; i++) {
    std::cout << "Element " << i << ": " << arr[i] << std::endl;
  }
  
  return 0;
}