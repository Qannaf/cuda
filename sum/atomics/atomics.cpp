#include <atomic>
#include <iostream>
#include <thread>

using namespace std;

void work(atomic<int> &data) {
  for (int i = 0; i < 10; i++) data++;
}

int main() {
  
  atomic<int> data(0);

  thread t0([&data]() { work(data); });
  thread t1([&data]() { work(data); });
  thread t2([&data]() { work(data); });
  thread t3([&data]() { work(data); });

  t0.join();
  t1.join();
  t2.join();
  t3.join();

  cout << "valeur finale = : " << data << '\n';

  return 0;
}
