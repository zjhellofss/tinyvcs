//
// Created by fss on 22-6-15.
//

#ifndef TINYVCS_INCLUDE_TICK_H_
#define TINYVCS_INCLUDE_TICK_H_
#define TICK(tag)  auto tag##_start = std::chrono::steady_clock::now(),tag##_end = tag##_start;
#define TOCK(tag) printf("%s costs %ld ms\n",#tag,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tag##_start).count());
#define TOCK_BATCH(tag, batch) printf("%s costs %ld ms\n",#tag,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tag##_start).count()/batch);

#endif //TINYVCS_INCLUDE_TICK_H_
