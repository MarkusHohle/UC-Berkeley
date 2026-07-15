#include <iostream>

int main() {
  int inside_loop;

  for (int i=0; i<10; i++) {
    inside_loop = i*2;
  }

  std::cout << "Value of inside loop " << inside_loop << std::endl;
  return 0;
}