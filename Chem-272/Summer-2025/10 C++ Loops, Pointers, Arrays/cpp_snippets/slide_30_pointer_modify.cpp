#include <iostream>

int main() {
  int x = 42;  
  int *ptr = &x; 
  *ptr = *ptr + 5; 
  std::cout << "New value of x: " << x << std::endl; 
  return 0;
}