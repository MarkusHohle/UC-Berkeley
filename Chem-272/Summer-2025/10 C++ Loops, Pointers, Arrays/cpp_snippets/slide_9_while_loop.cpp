#include <iostream>

void say_hello(std::string name) {
  std::cout << "Hello, " << name << "!" << std::endl;
}

int main(void) {
  int i = 0;
  while(i < 10) {
    say_hello("Dr. Nash");
    i++;
  }
  return 0;
}