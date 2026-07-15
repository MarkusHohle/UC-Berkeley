#include <iostream>

int main() {
  int x = 42;   // A normal integer variable
  int *ptr = &x; // Pointer storing the address of x

  std::cout << "Value of x: " << x << std::endl;
  std::cout << "Address of x: " << ptr << std::endl;

  // This will print 42. We are “dereferencing” ptr with the *
  std::cout << "Value pointed to by ptr: " << *ptr << std::endl; 
  return 0;
}