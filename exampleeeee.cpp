// -----------example 1---------------------

// #include <iostream>
// #include <thread>

// auto test(int x){
//     std::cout<<"other thread!"<<std::endl;
//     std::cout<<"arg passed in: "<< x << std::endl;
    
// }

// int main()
// {
//     std::thread mythread(&test , 100);
//     mythread.join();
//     std::cout << "Hello World from main thread" << std::endl;
    
//     return 0;
// }

//-------------------example 2--------------------

// #include <iostream>
// #include <thread>


// int main()
// {

//     auto lambda=[](int x){
//         std::cout<<"other thread!"<<std::endl;
//         std::cout<<"arg passed in: "<< x << std::endl;        
//     };

//     std::thread mythread(lambda , 100);
//     mythread.join();
//     std::cout << "Hello World from main thread" << std::endl;
    
//     return 0;
// }


//-----------------example- 3-----------------------


// #include <iostream>
// #include <thread>
// #include <vector>


// int main()
// {

//     auto lambda=[](int x){
//         std::cout<<"other thread!"<< std::this_thread::get_id()<<std::endl;
//         std::cout<<"arg passed in: "<< x << std::endl;        
//     };

//     std::vector<std::thread> threads;
//     for (int i=0; i<10; i++){
//         threads.push_back(std::thread(lambda, i));
//         threads[i].join();
//     }
//     std::cout << "Hello World from main thread" << std::endl;
    
//     return 0;
// }


//-------------------example 4-------------------------

// #include <iostream>
// #include <thread>
// #include <vector>


// int main()
// {

//     auto lambda=[](int x){
//         std::cout<<"other thread!"<< std::this_thread::get_id()<<std::endl;
//         std::cout<<"arg passed in: "<< x << std::endl;        
//     };

//     std::vector<std::thread> threads;
//     for (int i=0; i<10; i++){
//         threads.push_back(std::thread(lambda, i));
//     }

//     for (int i=0; i<10; i++){
//         threads[i].join();
//     }
//     std::cout << "Hello World from main thread" << std::endl;
    
//     return 0;
// }

//-----------------example 5----------------------


// #include <iostream>
// #include <thread>
// #include <vector>


// int main()
// {

//     auto lambda=[](int x){
//         std::cout<<"other thread!"<< std::this_thread::get_id()<<std::endl;
//         std::cout<<"arg passed in: "<< x << std::endl;        
//     };

//     std::vector<std::jthread> threads;
//     for (int i=0; i<10; i++){
//         threads.push_back(std::jthread(lambda, i));
//     }

//     std::cout << "Hello World from main thread" << std::endl;
    
//     return 0;
// }


//------------------example -6 -----------------



// #include <iostream>
// #include <thread>
// #include <vector>


// static int shared_value =0;

// void shared_value_increment(){
//     shared_value++;
// }

// int main()
// {


//     std::vector<std::thread> threads;
//     for (int i=0; i<10000; i++){
//         threads.push_back(std::thread(shared_value_increment));
//     }

//     for (int i=0; i<10000; i++){
//         threads[i].join();
//     }
//     std::cout << "shared value: " << shared_value << std::endl;
    
//     return 0;
// }


//------------------------------------example 7--------------------------



// #include <iostream>
// #include <thread>
// #include <vector>
// #include <mutex>

// std::mutex glock;
// static int shared_value =0;

// void shared_value_increment(){
//     glock.lock();
//     shared_value++;
//     glock.unlock();
// }

// int main()
// {


//     std::vector<std::thread> threads;
//     for (int i=0; i<10000; i++){
//         threads.push_back(std::thread(shared_value_increment));
//     }

//     for (int i=0; i<10000; i++){
//         threads[i].join();
//     }
//     std::cout << "shared value: " << shared_value << std::endl;
    
//     return 0;
// }
 

//--------------example 8 --------------------------------------



// #include <iostream>
// #include <thread>
// #include <vector>
// #include <mutex>

// std::mutex glock;
// static int shared_value =0;

// void shared_value_increment(){
//     std::lock_guard<std::mutex> lockgaurd(glock);  //locks and unlocks code on its own like automatically 
//     shared_value++;
// }

// int main()
// {


//     std::vector<std::thread> threads;
//     for (int i=0; i<10000; i++){
//         threads.push_back(std::thread(shared_value_increment));
//     }

//     for (int i=0; i<10000; i++){
//         threads[i].join();
//     }
//     std::cout << "shared value: " << shared_value << std::endl;
    
//     return 0;
// }
 

//--------------------------------example -9---------------------------------

// #include <iostream>
// #include <thread>
// #include <vector>
// #include <mutex>

// static std::atomic<int> shared_value =0;

// void shared_value_increment(){
//     shared_value++;
// }

// int main()
// {


//     std::vector<std::thread> threads;
//     for (int i=0; i<10000; i++){
//         threads.push_back(std::thread(shared_value_increment));
//     }

//     for (int i=0; i<10000; i++){
//         threads[i].join();
//     }
//     std::cout << "shared value: " << shared_value << std::endl;
    
//     return 0;
// }


//------------------------example-10-------------------------------

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

std::mutex glock;
std::condition_variable gconvar;



int main(){

    int result = 0;
    bool notified = false;
    //reporting thread
    std::thread reporter([&] {
        std::unique_lock<std::mutex> lock(glock);
        if(!notified){
            gconvar.wait(lock);
        }
        std::cout<< "reporter, result is: "<< result<< std::endl;
    });

    //working thread
    std::thread worker([&] {
        std::unique_lock<std::mutex> lock(glock);
        //do our work because we have the  lock
        result = 42+2+4;
        notified = true;

        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::cout<<" workc over\n"<<std::endl;

        gconvar.notify_one();


    });

    reporter.join();
    worker.join();
    std::cout<< "program complete"<<std::endl;

    return 0;
}

// -----------------------------example 11----------------------------
