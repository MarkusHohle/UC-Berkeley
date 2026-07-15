#include <iostream>

void say_hello(std::string name) {
  std::cout << "Hello, " << name << "!" << std::endl;
}

int main(void) {
  for(int i = 0; i < 10; i++) {
    say_hello("MSSE Student");
  }
  return 0;
}