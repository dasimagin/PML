#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>



class CsvWhitespace: public std::ctype<char> {
public:
    static const mask* getTable()
    {
        static std::vector<mask> table = makeTable();
        return &table[0];
    }

    CsvWhitespace(std::size_t refs = 0): ctype(getTable(), false, refs) {}

private:
    static std::vector<mask> makeTable()
    {
        // make a copy of the "C" locale table
        std::vector<mask> table(classic_table(), classic_table() + table_size);
        table[','] |=  space;  // comma will be classified as whitespace

        return table;
    }
};



using ID = uint64_t;
using Ids = std::vector<ID>;
using IdSet = std::set<ID>;

using Accuracy = uint16_t;
using Timestamp = uint32_t;

struct Point {
    float x;
    float y;
    Accuracy accuracy;
    Timestamp timestamp;
};

using Points = std::vector<Point>;
using PlaceIdToPoints = std::unordered_map<ID, Points>;

std::istream& operator>>(std::istream& in, Point& point)
{
    return in >> point.x >> point.y >> point.accuracy >> point.timestamp;
}

PlaceIdToPoints readTrain(std::istream& in)
{
    std::string header;
    std::getline(in, header);

    PlaceIdToPoints result;

    ID placeId;
    for (Point point; in >> point >> placeId; ) {
        result[placeId].emplace_back(point);
    }

    return result;
}

PlaceIdToPoints readTrain(const std::string& path)
{
    std::ifstream in(path);
    in.imbue(std::locale(in.getloc(), new CsvWhitespace));

    return readTrain(in);
}

Points readTest(std::istream& in)
{
    std::string header;
    std::getline(in, header);

    Points result;

    for (Point point; in >> point; ) {
        result.push_back(point);
    }

    return result;
}

Points readTest(const std::string& path)
{
    std::ifstream in(path);
    in.imbue(std::locale(in.getloc(), new CsvWhitespace));

    return readTest(in);
}




constexpr float PI = M_PI;
constexpr float SQRT2PI = std::sqrt(2 * PI);

class XDistribution {
public:
    XDistribution(float window): window_(window), values_() {}

    float prob(const Point& point) const;

    XDistribution& fit(const Points& points);

private:
    float window_;
    std::vector<float> values_;
};

XDistribution& XDistribution::fit(const Points& points)
{
    for (const auto& point: points) {
        values_.push_back(point.x);
    }

    std::sort(values_.begin(), values_.end());

    return *this;
}

float epanchenkov(float x)
{
    return 3 * (1 - x*x) / 4;
}

float XDistribution::prob(const Point& point) const
{
    float sum = 0.0;

    auto it = std::lower_bound(
        values_.begin(), values_.end(),
        point.x - window_
    );

    for (; it != values_.end(); ++it) {
        const float diff = *it - point.x;

        if (diff > window_) {
            break;
        }

        sum += epanchenkov(diff / window_);
    }

    return sum / (values_.size() * window_);
}

class YDistribution {
public:
    float prob(const Point& point) const;

    YDistribution& fit(const Points& points);

private:
    float mean_;
    float std_;
};

float YDistribution::prob(const Point& point) const
{
    const float diff = (point.y - mean_) / std_;
    return std::exp(-diff*diff / 2) / (std_ * SQRT2PI);
}

YDistribution& YDistribution::fit(const Points& points)
{
    float sum = 0.0;
    for (const auto& point: points) {
        sum += point.y;
    }

    mean_ = sum / points.size();

    float var = 0.0;
    for (const auto& point: points) {
        const float diff = point.y - mean_;
        var += diff * diff;
    }

    std_ = std::sqrt(var / points.size());

    return *this;
}

class TimestampDistribution {

public:
    using Hist = std::vector<float>;

    static constexpr uint64_t MINUTES_PER_WEEK = 7 * 24 * 60;
    static constexpr uint64_t WINDOW = 180;

    TimestampDistribution(): hist_(MINUTES_PER_WEEK / WINDOW, 0.0)
    {}

    TimestampDistribution& fit(const Points& points);

    float prob(const Point& point) const
    {
        return hist_[index(point)];
    }

private:
    uint64_t index(const Point& point) const
    {
        return (point.timestamp % MINUTES_PER_WEEK) /  WINDOW;
    }

    Hist hist_;
};

TimestampDistribution& TimestampDistribution::fit(const Points& points)
{
    const float step = 1.0 / points.size();

    for (const auto& point: points) {
        hist_[index(point)] += step;
    }

    return *this;
}


struct Distributions {
    float placeProb;
    XDistribution x;
    YDistribution y;
    TimestampDistribution timestamp;
};

using IndexPoint = boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian>;
using IndexBox = boost::geometry::model::box<IndexPoint>;
using Pair = std::pair<IndexPoint, ID>;
using Pairs = std::vector<Pair>;

using Index = boost::geometry::index::rtree<Pair, boost::geometry::index::quadratic<16>>;

Index buildIndex(const PlaceIdToPoints& placeIdToPoints)
{
    Index index;

    for (const auto& pair: placeIdToPoints) {
        const ID placeId = pair.first;
        const Points& points = pair.second;

        for (const auto& point: points) {
            index.insert(
                std::make_pair(IndexPoint(point.x, point.y), placeId)
            );
        }
    }

    return index;
}

using PlaceIdToDistributions = std::unordered_map<ID, Distributions>;

class NaiveBayesianClassifier {

public:
    NaiveBayesianClassifier& fit(const PlaceIdToPoints& placeIdToPoints);

    Ids predict(const Point& point, uint64_t optionN=3) const;

private:
    PlaceIdToDistributions placeIdToDistributions_;
    Index index_;
};

constexpr uint64_t NEAREST_POINT_N = 1000;

Ids NaiveBayesianClassifier::predict(const Point& point, uint64_t optionN) const
{
    constexpr float EPS = 1e-6;

    std::map<float, ID> scoreToPlaceId;

    Pairs pairs;
    index_.query(
        boost::geometry::index::nearest(IndexPoint(point.x, point.y), NEAREST_POINT_N),
        std::back_inserter(pairs)
    );

    IdSet placeIdSet;
    for (const auto& pair: pairs) {
        placeIdSet.insert(pair.second);
    }

    for (const auto& placeId: placeIdSet) {
        const Distributions& distributions = placeIdToDistributions_.at(placeId);

        if (distributions.placeProb < EPS) {
            continue;
        }
        float score = std::log(distributions.placeProb);

        const float xProb = distributions.x.prob(point);
        if (xProb < EPS) {
            continue;
        }
        score += std::log(xProb);

        const float yProb = distributions.y.prob(point);
        if (yProb < EPS) {
            continue;
        }
        score += std::log(yProb);

        const float timestampProb = distributions.timestamp.prob(point);
        if (timestampProb < EPS) {
            continue;
        }
        score += std::log(timestampProb);

        scoreToPlaceId.emplace(score, placeId);
    }

    Ids resultIds;
    for (auto it = scoreToPlaceId.rbegin(); it != scoreToPlaceId.rend(); ++it) {
        if (not optionN) {
            break;
        }

        resultIds.push_back(it->second);
        optionN--;
    }

    return resultIds;
}

using Predict = std::vector<Ids>;

constexpr float xWindow = 0.03;

PlaceIdToDistributions buildDistributions(const PlaceIdToPoints& placeIdToPoints)
{
    uint64_t totalPointN = 0;
    for (const auto& pair: placeIdToPoints) {
        totalPointN += pair.second.size();
    }

    PlaceIdToDistributions result;

    for (const auto& pair: placeIdToPoints) {
        const ID placeId = pair.first;
        const Points& points = pair.second;

        const float placeProb = static_cast<float>(points.size()) / totalPointN;

        result.emplace(
            placeId,
            Distributions {
                placeProb,
                XDistribution(xWindow).fit(points),
                YDistribution().fit(points),
                TimestampDistribution().fit(points)
            }
        );
    }

    return result;
}

NaiveBayesianClassifier& NaiveBayesianClassifier::fit(const PlaceIdToPoints& placeIdToPoints)
{
    auto placeIdToDistributions = std::async(
        std::launch::async,
        buildDistributions,
        placeIdToPoints
    );

    auto index = std::async(
        std::launch::async,
        buildIndex,
        placeIdToPoints
    );

    placeIdToDistributions_ = std::move(placeIdToDistributions.get());
    index_ = std::move(index.get());

    return *this;
}

Predict makePredict(
        const NaiveBayesianClassifier& classifier,
        const Points& points,
        uint64_t threadN)
{

    const uint64_t step = std::ceil(float(points.size()) / threadN);

    auto begin = points.begin();

    std::vector<std::future<Predict>> futures;
    for (uint64_t i = 0; i < threadN; ++i) {
        auto end = std::min(begin + step, points.end());

        futures.emplace_back(
            std::async(
                std::launch::async,
                [=](){
                    Predict predict;

                    for (auto it = begin; it != end; ++it) {
                        predict.push_back(classifier.predict(*it));
                    }

                    return predict;
                }
            )
        );

        begin = end;
    }

    Predict predict;
    for (auto& future: futures) {
        const Predict part = future.get();

        predict.insert(predict.end(), part.begin(), part.end());
    }

    return predict;
}

void write(std::ostream& out, const Predict& predict)
{
    out << "id,place_id" << std::endl;

    uint64_t id = 1;
    for (const auto& placeIds: predict) {
        out << id << ',';
        for (auto placeId: placeIds) {
            out << placeId << ' ';
        }
        out << std::endl;

        id++;
    }
}

void write(const std::string& path, const Predict& predict)
{
    std::ofstream out(path);
    write(out, predict);
}

int main() {
    NaiveBayesianClassifier classifier;

    { // train
        std::cout << "Read train dataset...";
        std::cout.flush();
        const PlaceIdToPoints placeIdToPoints = readTrain("./train.csv");
        std::cout << placeIdToPoints.size() << " classes" << std::endl;

        std::cout << "Fit model...";
        std::cout.flush();
        classifier.fit(placeIdToPoints);
        std::cout << "ok" << std::endl;
    }

    Predict predict;
    { // test
        std::cout << "Read test dataset...";
        std::cout.flush();
        const Points points = readTest("./test.csv");
        std::cout << "loaded " << points.size() << " points" << std::endl;

        const uint64_t threadN = std::thread::hardware_concurrency();
        std::cout << "Thread n = " << threadN << std::endl;

        std::cout << "Run prediction...";
        std::cout.flush();
        predict = makePredict(classifier, points, threadN);
        std::cout << "ok" << std::endl;
    }

    { // prepare result
        std::cout << "Print result...";
        std::cout.flush();
        write("predict.csv", predict);
        std::cout << "ok" << std::endl;
    }

    return 0;
}
