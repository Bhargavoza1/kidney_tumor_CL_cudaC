#pragma once
#include <vector>
#include <iostream>
#include <memory>
namespace Hex {
    template <typename T>
    class ITensor {
    public: 
        virtual ~ITensor() {}
        virtual void set(const std::vector<int>& indices, T value) = 0;
        virtual T get(const std::vector<int>& indices) const = 0;
        virtual void print() const = 0;
        virtual void setData(T* newData) = 0;
        virtual std::vector<int> getShape() const = 0;
        virtual const T* getData() const = 0;
        virtual T* getData() = 0;
    };

    template <typename T>
    class Tensor : public ITensor<T> {
    private:
        std::shared_ptr<T[]> data;
        std::vector<int> shape;
        bool _iscudafree;

    public:
        Tensor() : shape(std::vector<int>{}) {}
        Tensor(const std::vector<int>& shape ,  bool iscudafree = true);
        
        ~Tensor() override;
        void set(const std::vector<int>& indices, T value) override;
        T get(const std::vector<int>& indices) const override;
        void print() const override;
        void setData(T* newData) override;
        std::vector<int> getShape() const override;
        const T* getData() const override;
        T* getData() override;

        void cudafree();

        void printshape() const  ;
        void reshape(const std::vector<int>& new_shape); 
    private:
        int calculateIndex(const std::vector<int>& indices) const;
        void printHelper(const T* data, const std::vector<int>& shape, int dimension, std::vector<int> indices) const;
        std::vector<int> calculateIndices(int index) const;
    };
}
