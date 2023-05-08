#ifndef KOUEK_IMGUI_TF_H
#define KOUEK_IMGUI_TF_H

#include <type_traits>

#include <fstream>
#include <string>

#include <algorithm>
#include <tuple>
#include <array>
#include <vector>

#include <implot.h>

namespace kouek {

	template <typename VoxTy> class TFWidget {
	private:
		static constexpr size_t VTVar = std::numeric_limits<VoxTy>::max() -
			std::numeric_limits<VoxTy>::min() + 1;

		template <typename VoxTy, size_t VTVar> class TFKeyPoints {
		private:
			std::vector<double> xs;
			std::vector<double> ys;
			std::vector<ImVec4> cols;

		public:
			TFKeyPoints() { static_assert(std::is_integral_v<VoxTy>); }

			void Add(double key, const std::array<double, 4>& val,
				double err = 5.0) {
				if (key < 0 || key >= (double)VTVar)
					return;
				if (xs.size() == VTVar)
					return;

				auto xItr = std::find_if(xs.begin(), xs.end(), [&](double x) {
					return abs(round(x) - round(key)) <= err;
					});
				if (xItr != xs.end()) {
					auto offs = xItr - xs.begin();
					auto yItr = ys.begin();
					std::advance(yItr, offs);
					*yItr = val[3];
					auto cItr = cols.begin();
					std::advance(cItr, offs);
					*cItr =
						ImVec4{ (float)val[0], (float)val[1], (float)val[2], 1.f };
				}
				else {
					xs.emplace_back(key);
					ys.emplace_back(val[3]);
					cols.emplace_back(
						ImVec4{ (float)val[0], (float)val[1], (float)val[2], 1.f });
				}
			}
			inline void Delete(size_t idx) {
				{
					auto itr = xs.begin();
					std::advance(itr, idx);
					itr = xs.erase(itr);
				}
				{
					auto itr = ys.begin();
					std::advance(itr, idx);
					itr = ys.erase(itr);
				}
				{
					auto itr = cols.begin();
					std::advance(itr, idx);
					itr = cols.erase(itr);
				}
			}

			auto& GetXs() { return xs; }
			auto& GetYs() { return ys; }
			auto& GetCols() { return cols; }
		};

		bool flatTFNeedUpdate = true;
		size_t lastKPt = std::numeric_limits<size_t>::max();
		std::array<float, 4> lastKPtCol;
		std::array<double, 2> upBoundXY{ (double)VTVar, 1.0 };

		std::vector<double> orderedXs;
		std::vector<double> orderedYs;
		std::vector<size_t> orderedIndices;

		std::vector<float> flatTF;

		std::string bufferedTFPath;

		TFKeyPoints<VoxTy, VTVar> tfKPts;

	public:
		TFWidget() {
			tfKPts.Add((double)0, std::array{ 0.0, 0.0, 0.0, 0.0 });
			tfKPts.Add((double)VTVar - 1, std::array{ 1.0, 1.0, 1.0, 1.0 });
			orderedXs.emplace_back((double)0);
			orderedXs.emplace_back((double)VTVar - 1);
			orderedYs.emplace_back((double)0);
			orderedYs.emplace_back((double)1.0);
			orderedIndices.emplace_back(0);
			orderedIndices.emplace_back(1);

			flatTF.reserve(VTVar);
		}

		void Load(const std::string& path) {
			std::ifstream is(path, std::ios::in);
			if (!is.is_open())
				throw std::runtime_error("Cannot open file: " + path + " .");
			bufferedTFPath = path;

			std::string buf;
			while (std::getline(is, buf)) {
				double x;
				std::array<double, 4> col;

				auto cnt = sscanf(buf.c_str(), "%lf%lf%lf%lf%lf", &x, &col[0], &col[1], &col[2], &col[3]);
				if (cnt != 5)
					continue;

				for (auto& v : col)
					v /= 255.0;
				clamp(x, col[3]);

				tfKPts.Add(x, col);
			}

			is.close();
			reorder();
		}

		void Save(const std::string& path = "") {
			if (path.empty() && bufferedTFPath.empty())
				throw std::runtime_error("No specified file path to save transfer function.");

			std::ofstream os(path.empty() ? bufferedTFPath : path, std::ios::out);

			const auto& xs = tfKPts.GetXs();
			const auto& ys = tfKPts.GetYs();
			const auto& cols = tfKPts.GetCols();
			std::string buf;
			for (size_t i = 0; i < orderedIndices.size(); ++i) {
				if (i != 0)
					os << '\n';

				auto orderedIdx = orderedIndices[i];
				buf.clear();
				buf += std::to_string(xs[orderedIdx]);
				buf += ' ';
				buf += std::to_string(cols[orderedIdx].x * 255.0);
				buf += ' ';
				buf += std::to_string(cols[orderedIdx].y * 255.0);
				buf += ' ';
				buf += std::to_string(cols[orderedIdx].z * 255.0);
				buf += ' ';
				buf += std::to_string(ys[orderedIdx] * 255.0);

				os << buf;
			}

			os.close();
		}

		bool operator()() {
			ImGui::BulletText("CTRL + Left Click => Add a Key Point");
			ImGui::BulletText("Drag a Key Point => Move the Key Point");
			ImGui::BulletText("Left Click on a Key Point => Select");
			ImGui::BulletText("Left Click anywhere except a Key Point => Cancel "
				"previous selection");

			auto& xs = tfKPts.GetXs();
			auto& ys = tfKPts.GetYs();
			auto& cols = tfKPts.GetCols();

			bool kPtModified = false;
			if (ImPlot::BeginPlot("Transfer Function")) {
				auto num = xs.size();
				int CDPRet = 0;
				for (size_t i = 0; i < num; ++i) {
					if (auto ret =
						ImPlot::ClickDragPoint(i, xs.data() + i, ys.data() + i,
							cols[i], 4.f, 0, i == lastKPt);
						ret != 0) {
						CDPRet |= ret;
						lastKPt = i;
					}
					clamp(xs[i], ys[i]);
				}

				// 2 end points should be kept
				xs[0] = 0.0;
				xs[1] = (double)(VTVar - 1);

				if (CDPRet != 0) {
					lastKPtCol[0] = cols[lastKPt].x;
					lastKPtCol[1] = cols[lastKPt].y;
					lastKPtCol[2] = cols[lastKPt].z;
					lastKPtCol[3] = ys[lastKPt];

					kPtModified = (CDPRet & 0x2) != 0; // has been dragged?
				}

				if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0))
					if (ImGui::GetIO().KeyCtrl) {
						auto pt = ImPlot::GetPlotMousePos();
						clamp(pt.x, pt.y);
						tfKPts.Add(pt.x, std::array{ 1.0, 1.0, 1.0, pt.y });
						kPtModified = true;
					}
					else
						lastKPt = std::numeric_limits<size_t>::max();

				ImPlot::PlotLine("", orderedXs.data(), orderedYs.data(),
					orderedXs.size());
				static constexpr ImVec4 BOUND_COL{ .2f, .2f, .2f, 1.f };
				ImPlot::DragLineX(0, upBoundXY.data() + 0, BOUND_COL, 1.f,
					ImPlotDragToolFlags_NoInputs);
				ImPlot::DragLineY(1, upBoundXY.data() + 1, BOUND_COL, 1.f,
					ImPlotDragToolFlags_NoInputs);

				ImPlot::EndPlot();
			}


			if (ImGui::Button("Save"))
				Save();

			if (lastKPt != std::numeric_limits<size_t>::max()) {
				ImGui::SameLine();
				auto del = ImGui::Button("Delete") && lastKPt != 0 && lastKPt != 1;

				ImGui::SameLine();
				if (ImGui::ColorEdit4("##Color", lastKPtCol.data())) {
					cols[lastKPt].x = lastKPtCol[0];
					cols[lastKPt].y = lastKPtCol[1];
					cols[lastKPt].z = lastKPtCol[2];
					ys[lastKPt] = lastKPtCol[3];
					kPtModified = true;
				}

				ImGui::SameLine();
				ImGui::Text("x:%f", xs[lastKPt]);

				if (del) {
					tfKPts.Delete(lastKPt);
					lastKPt = std::numeric_limits<size_t>::max();
					kPtModified = true;
				}
			}

			if (kPtModified)
				reorder();

			return kPtModified;
		}
		inline const auto& GetFlatTF() {
			const auto& xs = tfKPts.GetXs();
			const auto& ys = tfKPts.GetYs();
			const auto& cols = tfKPts.GetCols();

			if (flatTFNeedUpdate) {
				flatTF.clear();
				VoxTy i = 0;
				auto x1 = orderedXs[0];
				auto x01 = x1;
				auto j0 = 0;
				auto j1 = orderedIndices[0];
				VoxTy s = 0;
				while (true) {
					if (s == (VoxTy)round(x1)) {
						++i;
						auto x0 = x1;
						if (i == orderedIndices.size())
							break;
						j0 = j1;
						j1 = orderedIndices[i];
						x1 = orderedXs[i];
						x01 = x1 - x0;
					}
					auto k = (x1 - (double)s) / x01;
					flatTF.emplace_back(
						(float)(k * cols[j0].x + (1 - k) * cols[j1].x));
					flatTF.emplace_back(
						(float)(k * cols[j0].y + (1 - k) * cols[j1].y));
					flatTF.emplace_back(
						(float)(k * cols[j0].z + (1 - k) * cols[j1].z));
					flatTF.emplace_back((float)(k * ys[j0] + (1 - k) * ys[j1]));

					++s;
				}
				auto j = orderedIndices.back();
				flatTF.emplace_back((float)(cols[j].x));
				flatTF.emplace_back((float)(cols[j].y));
				flatTF.emplace_back((float)(cols[j].z));
				flatTF.emplace_back((float)ys[j]);

				flatTFNeedUpdate = false;
			}
			return flatTF;
		}

	private:
		static inline void clamp(double& x, double& y) {
			if (x <= 0.0)
				x = 0.0;
			if (x >= (double)VTVar)
				x = (double)(VTVar - 1);
			if (y <= 0.0)
				y = 0.0;
			if (y > 1.0)
				y = 1.0;
		}
		inline void reorder() {
			auto& xs = tfKPts.GetXs();
			auto& ys = tfKPts.GetYs();
			auto _xs = xs;

			orderedIndices.clear();
			for (size_t i = 0; i < xs.size(); ++i)
				orderedIndices.emplace_back(i);
			std::sort(orderedIndices.begin(), orderedIndices.end(),
				[&](size_t a, size_t b) { return _xs[a] < _xs[b]; });
			orderedXs.clear();
			orderedYs.clear();
			for (size_t i = 0; i < xs.size(); ++i) {
				orderedXs.emplace_back(xs[orderedIndices[i]]);
				orderedYs.emplace_back(ys[orderedIndices[i]]);
			}

			flatTFNeedUpdate = true;
		}
	};

} // namespace kouek

#endif // !KOUEK_IMGUI_TF_H
