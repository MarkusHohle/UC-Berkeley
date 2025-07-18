#include <iostream>

int main() {
  int a = 10;
  int b = 20;
  int *ptr = &a;

  std::cout << "ptr points to a: " << *ptr << std::endl;

  ptr = &b; // ptr now points to b
  std::cout << "ptr now points to b: " << *ptr << std::endl;

  return 0;
}