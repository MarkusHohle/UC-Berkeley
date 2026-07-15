#include <iostream>

int main() {
  int size;
  std::cout << "Input a vector size" << std::endl;
  std::cin >> size;

  int *arr = new int[size];

  for (int i=0; i<size; i++) {
    arr[i] = i;
  }

  delete [] arr; 
  return 0;
}