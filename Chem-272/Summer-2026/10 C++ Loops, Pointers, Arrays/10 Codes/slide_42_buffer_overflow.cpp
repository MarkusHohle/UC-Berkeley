#include <iostream>

int main() {
  int arr[5] = {1, 2, 3, 4, 5};

  for (int i=0; i<7; i++) {
    arr[i] = arr[i] + 1;
    std::cout << i << " " << arr[i] << std::endl;
  }
  return 0;
}